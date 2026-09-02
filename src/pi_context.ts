import type { AssistantMessage, Context, Tool } from "@mupt-ai/pi-ai";

import { RouterFrameworkError } from "./framework_error.js";
import { encodeOpenAIReasoningSignature } from "./pi_reasoning_signature.js";
import { isRecord } from "./types.js";
import type {
  RouterContent,
  RouterProviderIdentity,
  RouterRequest,
} from "./framework_types.js";
import {
  compatibleProviderContinuation,
  isOpenAIResponsesApi,
  type ProviderContinuationState,
} from "./continuation_state.js";
import type { PiModel } from "./pi_types.js";

export function piContext(request: RouterRequest, model: PiModel): Context {
  const system: string[] = [];
  const messages: Context["messages"] = [];
  const calls = new Map<string, string>();
  const timestamp = Date.now();
  let assistant: Extract<Context["messages"][number], { role: "assistant" }> | undefined;
  let sawConversationItem = false;

  const flushAssistant = () => {
    if (assistant === undefined) return;
    if (assistant.content.length === 0) assistant.content.push({ type: "text", text: "" });
    assistant.stopReason = assistant.content.some((part) => part.type === "toolCall")
      ? "toolUse"
      : "stop";
    messages.push(assistant);
    assistant = undefined;
  };
  const currentAssistant = () => {
    assistant ??= {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: zeroPiUsage(),
      stopReason: "stop",
      timestamp,
    };
    return assistant;
  };

  for (const item of request.items) {
    if (item.type === "message" && (item.role === "system" || item.role === "developer")) {
      flushAssistant();
      const text = systemText(item.content);
      if (!text) continue;
      if (!sawConversationItem) {
        // Leading preamble is identical on every request, so folding it into
        // the system prompt keeps the provider payload prefix stable.
        system.push(text);
        continue;
      }
      // Mid-conversation system/developer notes arrive between turns (Claude
      // Code appends a `<total_tokens>` reminder after every tool result).
      // Folding them into the system prompt mutates the front of the provider
      // payload, so prefix-matching prompt caches miss and re-write the whole
      // conversation at cache-write prices. Keep them positional instead.
      appendPositionalNote(messages, text, timestamp);
      continue;
    }
    sawConversationItem = true;
    if (item.type === "message" && item.role === "user") {
      flushAssistant();
      appendUserMessage(messages, piContent(item.content, "user message"), timestamp);
      continue;
    }
    if (item.type === "message") {
      const pending = currentAssistant();
      pending.content.push(...item.content.map((part) => {
        if (part.type !== "text") {
          throw new RouterFrameworkError(
            "invalid_request",
            "Pi does not support image content in assistant messages.",
            "pi_assistant_image_unsupported",
          );
        }
        return { type: "text" as const, text: part.text };
      }));
      continue;
    }
    if (item.type === "tool_call") {
      const argumentsValue = toolArguments(item.arguments, item.id);
      currentAssistant().content.push({
        type: "toolCall",
        id: item.id,
        name: item.name,
        arguments: argumentsValue,
      });
      calls.set(item.id, item.name);
      continue;
    }
    if (item.type === "reasoning") {
      const target: RouterProviderIdentity = {
        provider: model.provider,
        api: model.api,
        model: model.id,
      };
      const continuation = compatibleProviderContinuation(item.continuation, target);
      if (continuation) {
        currentAssistant().content.push(thinkingContent(item, continuation));
      } else {
        const readable = [...(item.summary ?? []), ...item.content].join("\n");
        if (readable) {
          // Cross-provider: the encrypted blob is not replayable, so carry the
          // readable reasoning as plain text instead of dropping the prior
          // turn's chain-of-thought entirely.
          currentAssistant().content.push({ type: "text", text: readable });
        }
      }
      continue;
    }
    if (item.type === "hosted_tool_call") {
      if (!isOpenAIResponsesApi(model.api)) {
        // Dropping the item would silently lose a conversation turn.
        throw new RouterFrameworkError(
          "configuration",
          `Hosted tool calls are not representable on the '${model.api}' API; route hosted-tool conversations to a Responses-API model.`,
          "hosted_tool_call_unrepresentable",
        );
      }
      (currentAssistant().content as unknown as Array<Record<string, unknown>>).push({
        type: "providerItem",
        item: item.payload,
      });
      continue;
    }
    flushAssistant();
    messages.push({
      role: "toolResult",
      toolCallId: item.toolCallId,
      toolName: calls.get(item.toolCallId) ?? "tool",
      content: piContent(item.content, `tool result '${item.toolCallId}'`),
      isError: item.isError ?? false,
      timestamp,
    });
  }
  flushAssistant();

  const tools: Tool[] = request.tools.map((tool) => ({
    name: tool.name,
    description: tool.description ?? "",
    parameters: tool.inputSchema as Tool["parameters"],
  }));
  return {
    ...(system.length === 0 ? {} : { systemPrompt: system.join("\n\n") }),
    messages,
    ...(tools.length === 0 ? {} : { tools }),
  };
}

function thinkingContent(
  item: Extract<RouterRequest["items"][number], { type: "reasoning" }>,
  continuation: ProviderContinuationState,
): { type: "thinking"; thinking: string; thinkingSignature?: string; redacted?: boolean } {
  if (continuation.kind === "openai_reasoning") {
    const providerItemId = continuation.providerItemId ?? item.id;
    return {
      type: "thinking",
      thinking: [...(item.summary ?? []), ...item.content].join("\n"),
      thinkingSignature: encodeOpenAIReasoningSignature({
        ...(providerItemId ? { id: providerItemId } : {}),
        summary: item.summary ?? [],
        content: item.content,
        encryptedContent: continuation.encryptedContent,
      }),
    };
  }
  if (continuation.kind === "anthropic_thinking") {
    return {
      type: "thinking",
      thinking: continuation.thinking,
      thinkingSignature: continuation.signature,
    };
  }
  return {
    type: "thinking",
    thinking: "",
    thinkingSignature: continuation.data,
    redacted: true,
  };
}

function appendPositionalNote(
  messages: Context["messages"],
  text: string,
  timestamp: number,
): void {
  const previous = messages.at(-1);
  if (previous?.role === "toolResult") {
    previous.content = [...piContentBlocks(previous.content), { type: "text", text }];
    return;
  }
  appendUserMessage(messages, [{ type: "text", text }], timestamp);
}

function appendUserMessage(
  messages: Context["messages"],
  content: Extract<Context["messages"][number], { role: "user" }>["content"],
  timestamp: number,
): void {
  const previous = messages.at(-1);
  if (previous?.role === "user") {
    previous.content = [...piContentBlocks(previous.content), ...piContentBlocks(content)];
    return;
  }
  messages.push({ role: "user", content, timestamp });
}

function piContentBlocks(
  content: Extract<Context["messages"][number], { role: "user" | "toolResult" }>["content"],
): Exclude<Extract<Context["messages"][number], { role: "user" }>["content"], string> {
  return typeof content === "string" ? [{ type: "text", text: content }] : content;
}

function systemText(content: readonly RouterContent[]): string {
  return content.map((part) => {
    if (part.type !== "text") {
      throw new RouterFrameworkError(
        "invalid_request",
        "Pi does not support image content in system or developer messages.",
        "pi_system_image_unsupported",
      );
    }
    return part.text;
  }).join("");
}

function piContent(
  content: readonly RouterContent[],
  location: string,
): Array<{ type: "text"; text: string } | { type: "image"; data: string; mimeType: string }> {
  return content.map((part) => {
    if (part.type === "text") return { type: "text", text: part.text };
    const image = dataImage(part.url);
    if (image === null) {
      throw new RouterFrameworkError(
        "invalid_request",
        `Pi ${location} images must use base64 data URLs.`,
        "pi_remote_image_unsupported",
      );
    }
    return { type: "image", data: image.data, mimeType: image.mimeType };
  });
}

function dataImage(url: string): { data: string; mimeType: string } | null {
  const match = /^data:(image\/[a-z0-9.+-]+);base64,([a-z0-9+/=\s]+)$/i.exec(url);
  if (match === null) return null;
  return { mimeType: match[1]!, data: match[2]!.replace(/\s/g, "") };
}

function toolArguments(value: string | Record<string, unknown>, id: string): Record<string, unknown> {
  if (typeof value !== "string") return value;
  // Providers emit "" for zero-argument tool calls; replay it as {}.
  if (value === "") return {};
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch (error) {
    throw new RouterFrameworkError(
      "invalid_request",
      `Tool call '${id}' arguments must be valid JSON.`,
      "pi_tool_arguments_invalid",
      undefined,
      { cause: error },
    );
  }
  if (!isRecord(parsed)) {
    throw new RouterFrameworkError(
      "invalid_request",
      `Tool call '${id}' arguments must be a JSON object.`,
      "pi_tool_arguments_invalid",
    );
  }
  return parsed;
}

function zeroPiUsage(): AssistantMessage["usage"] {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}
