import type {
  AssistantMessage,
  AssistantMessageEvent,
  ThinkingContent,
  ToolCall,
} from "@mupt-ai/pi-ai";

import { RouterFrameworkError } from "./framework_error.js";
import { decodeOpenAIReasoningSignature } from "./pi_reasoning_signature.js";
import { isRecord } from "./types.js";
import {
  isOpenAIResponsesApi,
  type ProviderContinuationState,
  type ProviderIdentity,
} from "./continuation_state.js";
import {
  isRouterHostedToolCallStatus,
  type RouterCompletion,
  type RouterOutputHostedToolCall,
  type RouterOutputReasoning,
  type RouterStreamEvent,
} from "./framework_types.js";

export function routerCompletion(
  message: AssistantMessage,
  signal: AbortSignal,
): RouterCompletion {
  assertPiSuccess(message, signal);
  const content: RouterCompletion["content"] = [];
  const source = providerIdentity(message);
  for (const part of message.content) {
    if (part.type === "text") content.push({ type: "text", text: part.text });
    if (part.type === "thinking") content.push(routerReasoning(part, source));
    if (part.type === "toolCall") {
      content.push({
        type: "tool_call",
        id: part.id,
        name: part.name,
        arguments: part.arguments,
      });
    }
    const hostedItem = providerItemRecord(part);
    if (hostedItem !== null) content.push(hostedToolCall(hostedItem, source));
  }
  return {
    ...(message.responseId === undefined ? {} : { id: message.responseId }),
    createdAtMs: message.timestamp,
    content,
    finishReason: routerFinishReason(message.stopReason),
    usage: routerUsage(message),
  };
}

export async function* routerEvents(
  events: AsyncIterable<AssistantMessageEvent>,
  signal: AbortSignal,
): AsyncIterable<RouterStreamEvent> {
  const text = new Map<number, string>();
  const thinking = new Map<number, string>();
  const endedThinking = new Set<number>();
  const openTools = new Map<number, { receivedDelta: boolean }>();
  const completedTools = new Map<number, Pick<ToolCall, "id" | "name">>();
  for await (const event of events) {
    if (event.type === "text_start") {
      text.set(event.contentIndex, text.get(event.contentIndex) ?? "");
      continue;
    }
    if (event.type === "text_delta") {
      if (event.delta.length > 0) {
        text.set(event.contentIndex, (text.get(event.contentIndex) ?? "") + event.delta);
        yield { type: "text_delta", index: event.contentIndex, delta: event.delta };
      }
      continue;
    }
    if (event.type === "text_end") {
      const suffix = remainingText(text.get(event.contentIndex) ?? "", event.content);
      if (suffix.length > 0) yield { type: "text_delta", index: event.contentIndex, delta: suffix };
      text.set(event.contentIndex, event.content);
      continue;
    }
    if (event.type === "thinking_start") {
      thinking.set(event.contentIndex, "");
      continue;
    }
    if (event.type === "thinking_delta") {
      if (event.delta.length > 0) {
        thinking.set(event.contentIndex, (thinking.get(event.contentIndex) ?? "") + event.delta);
        yield { type: "reasoning_delta", index: event.contentIndex, delta: event.delta };
      }
      continue;
    }
    if (event.type === "thinking_end") {
      const streamed = thinking.get(event.contentIndex) ?? "";
      const suffix = remainingText(streamed, event.content);
      if (suffix.length > 0) yield { type: "reasoning_delta", index: event.contentIndex, delta: suffix };
      thinking.set(event.contentIndex, event.content);
      endedThinking.add(event.contentIndex);
      const part = event.partial.content[event.contentIndex];
      const source = providerIdentity(event.partial);
      if (part?.type === "thinking") {
        const reasoning = routerReasoning(part, source);
        yield {
          type: "reasoning_end",
          index: event.contentIndex,
          ...(part.redacted === true ? { redacted: true } : {}),
          source,
          ...(reasoning.continuation ? { continuation: reasoning.continuation } : {}),
        };
      } else {
        yield { type: "reasoning_end", index: event.contentIndex, source };
      }
      continue;
    }
    if (event.type === "toolcall_start") {
      const call = event.partial.content[event.contentIndex];
      if (call?.type !== "toolCall") throw invalidPiStream("tool-call start is missing its tool call.");
      openTools.set(event.contentIndex, { receivedDelta: false });
      yield {
        type: "tool_call_start",
        index: event.contentIndex,
        id: call.id,
        name: call.name,
      };
      continue;
    }
    if (event.type === "toolcall_delta") {
      const state = openTools.get(event.contentIndex);
      if (state === undefined) throw invalidPiStream("tool-call delta arrived before its start.");
      if (event.delta.length > 0) {
        state.receivedDelta = true;
        yield { type: "tool_call_delta", index: event.contentIndex, delta: event.delta };
      }
      continue;
    }
    if (event.type === "toolcall_end") {
      const state = openTools.get(event.contentIndex);
      if (state === undefined) {
        yield* completeToolCall(event.contentIndex, event.toolCall);
        completedTools.set(event.contentIndex, event.toolCall);
        continue;
      }
      if (!state.receivedDelta) {
        yield {
          type: "tool_call_delta",
          index: event.contentIndex,
          delta: JSON.stringify(event.toolCall.arguments),
        };
      }
      openTools.delete(event.contentIndex);
      completedTools.set(event.contentIndex, event.toolCall);
      yield { type: "tool_call_end", index: event.contentIndex };
      continue;
    }
    if (event.type === "done") {
      assertPiSuccess(event.message, signal);
      if (event.reason !== "stop" && event.reason !== "length" && event.reason !== "toolUse") {
        throw invalidPiStream(`stream ended with invalid reason '${String(event.reason)}'.`);
      }
      if (openTools.size > 0) throw invalidPiStream("stream finished with open tool calls.");
      yield* reconcileTerminalOutput(event.message, text, thinking, endedThinking, completedTools);
      yield {
        type: "finish",
        finishReason: routerFinishReason(event.reason),
        usage: routerUsage(event.message),
      };
      return;
    }
    if (event.type === "error") {
      if (event.reason === "aborted") throw piCancellation(event.error, signal);
      throw piProviderError(event.error.errorMessage ?? "Pi provider execution failed.");
    }
  }
  throw invalidPiStream("stream ended without a terminal event.");
}

function* reconcileTerminalOutput(
  message: AssistantMessage,
  text: ReadonlyMap<number, string>,
  thinking: ReadonlyMap<number, string>,
  endedThinking: ReadonlySet<number>,
  completedTools: ReadonlyMap<number, Pick<ToolCall, "id" | "name">>,
): Iterable<RouterStreamEvent> {
  for (const index of text.keys()) {
    if (message.content[index]?.type !== "text") {
      throw invalidPiStream(`terminal output changed text block ${index}.`);
    }
  }
  for (const index of thinking.keys()) {
    if (message.content[index]?.type !== "thinking") {
      throw invalidPiStream(`terminal output changed thinking block ${index}.`);
    }
  }
  for (const [index, streamed] of completedTools) {
    const terminal = message.content[index];
    if (terminal?.type !== "toolCall" || terminal.id !== streamed.id || terminal.name !== streamed.name) {
      throw invalidPiStream(`terminal output changed tool call ${index}.`);
    }
  }

  for (const [index, part] of message.content.entries()) {
    if (part.type === "text") {
      const suffix = remainingText(text.get(index) ?? "", part.text);
      if (suffix.length > 0) yield { type: "text_delta", index, delta: suffix };
      continue;
    }
    if (part.type === "thinking") {
      // Started-but-unended thinking blocks reconcile like unended text:
      // emit the remaining delta, then the proper reasoning_end.
      if (!endedThinking.has(index)) {
        yield* completeReasoning(index, part, providerIdentity(message), thinking.get(index) ?? "");
      }
      continue;
    }
    if (part.type === "toolCall") {
      if (!completedTools.has(index)) yield* completeToolCall(index, part);
      continue;
    }
    const hostedItem = providerItemRecord(part);
    if (hostedItem !== null) {
      yield { ...hostedToolCall(hostedItem, providerIdentity(message)), index };
    }
  }
}

function* completeToolCall(index: number, call: ToolCall): Iterable<RouterStreamEvent> {
  yield { type: "tool_call_start", index, id: call.id, name: call.name };
  yield { type: "tool_call_delta", index, delta: JSON.stringify(call.arguments) };
  yield { type: "tool_call_end", index };
}

function* completeReasoning(
  index: number,
  part: ThinkingContent,
  source: ProviderIdentity,
  streamed = "",
): Iterable<RouterStreamEvent> {
  const suffix = remainingText(streamed, part.thinking);
  if (suffix.length > 0) {
    yield { type: "reasoning_delta", index, delta: suffix };
  }
  const reasoning = routerReasoning(part, source);
  yield {
    type: "reasoning_end",
    index,
    ...(part.redacted === true ? { redacted: true } : {}),
    source,
    ...(reasoning.continuation ? { continuation: reasoning.continuation } : {}),
  };
}

function providerItemRecord(part: unknown): Record<string, unknown> | null {
  if (!isRecord(part) || part.type !== "providerItem") return null;
  if (!isRecord(part.item)) {
    throw unsupportedHostedTool("the provider item has no payload record");
  }
  return part.item;
}

function hostedToolCall(
  item: Record<string, unknown>,
  source: ProviderIdentity,
): RouterOutputHostedToolCall {
  if (item.type !== "web_search_call") {
    throw unsupportedHostedTool(
      `only web_search_call items are replayable, got '${String(item.type)}'`,
    );
  }
  if (typeof item.id !== "string" || item.id.length === 0) {
    throw unsupportedHostedTool("the web_search_call item has no id");
  }
  const status = hostedToolCallStatus(item.status);
  return {
    type: "hosted_tool_call",
    id: item.id,
    tool: "web_search",
    providerType: "web_search_call",
    ...(status === undefined ? {} : { status }),
    payload: item,
    source,
  };
}

function hostedToolCallStatus(value: unknown): RouterOutputHostedToolCall["status"] {
  if (value === undefined || isRouterHostedToolCallStatus(value)) return value;
  throw unsupportedHostedTool(`web_search_call status '${String(value)}' is not replayable`);
}

function unsupportedHostedTool(detail: string): RouterFrameworkError {
  return new RouterFrameworkError(
    "executor",
    `Pi returned a hosted tool item the router cannot replay: ${detail}.`,
    "pi_hosted_tool_unsupported",
  );
}

function remainingText(streamed: string, terminal: string): string {
  if (!terminal.startsWith(streamed)) {
    throw invalidPiStream("terminal text does not extend the streamed text.");
  }
  return terminal.slice(streamed.length);
}

function providerIdentity(
  message: Pick<AssistantMessage, "provider" | "api" | "model">,
): ProviderIdentity {
  return { provider: message.provider, api: message.api, model: message.model };
}

function routerReasoning(
  part: ThinkingContent,
  source: ProviderIdentity,
): RouterOutputReasoning {
  const signature = typeof part.thinkingSignature === "string" && part.thinkingSignature.length > 0
    ? part.thinkingSignature
    : undefined;
  let continuation: ProviderContinuationState | undefined;
  if (isOpenAIResponsesApi(source.api) && signature) {
    const native = decodeOpenAIReasoningSignature(signature);
    if (native?.encryptedContent) {
      continuation = {
        kind: "openai_reasoning",
        source,
        encryptedContent: native.encryptedContent,
        ...(native.id ? { providerItemId: native.id } : {}),
      };
    }
  } else if (source.api === "anthropic-messages" && signature) {
    continuation = part.redacted === true
      ? { kind: "anthropic_redacted_thinking", source, data: signature }
      : { kind: "anthropic_thinking", source, thinking: part.thinking, signature };
  }
  return {
    type: "reasoning",
    text: part.thinking,
    ...(part.redacted === true ? { redacted: true } : {}),
    source,
    ...(continuation ? { continuation } : {}),
  };
}

function assertPiSuccess(message: AssistantMessage, signal: AbortSignal): void {
  if (message.stopReason === "error") {
    throw piProviderError(message.errorMessage ?? "Pi provider execution failed.");
  }
  if (message.stopReason === "aborted") throw piCancellation(message, signal);
}

function routerFinishReason(
  reason: "stop" | "length" | "toolUse" | AssistantMessage["stopReason"],
): RouterCompletion["finishReason"] {
  if (reason === "length") return "length";
  if (reason === "toolUse") return "tool_calls";
  return "stop";
}

function routerUsage(
  message: Pick<AssistantMessage, "usage">,
): NonNullable<RouterCompletion["usage"]> {
  return {
    inputTokens: message.usage.input,
    outputTokens: message.usage.output,
    cacheReadTokens: message.usage.cacheRead,
    cacheWriteTokens: message.usage.cacheWrite,
    totalTokens: message.usage.totalTokens,
  };
}

function piCancellation(
  message: Pick<AssistantMessage, "errorMessage">,
  signal: AbortSignal,
): RouterFrameworkError {
  return new RouterFrameworkError(
    "cancelled",
    typeof signal.reason === "string"
      ? signal.reason
      : message.errorMessage ?? "Request cancelled.",
    "request_cancelled",
  );
}

function piProviderError(message: string): RouterFrameworkError {
  return new RouterFrameworkError("executor", message, "pi_provider_error");
}

function invalidPiStream(message: string): RouterFrameworkError {
  return new RouterFrameworkError("executor", `Pi ${message}`, "pi_stream_invalid");
}
