// Group Scene Director
// Makes an OpenAI-compatible "director" model pick the next group-chat speaker
// via a forced function tool call, and lets SillyTavern run the actual generation.
//
// Modes:
//  - chained (default): pick first speaker -> ST generates -> probe "who next / end" -> repeat
//  - batch: pick the whole turn's roster up front -> ST generates everyone in order
//
// The director request is a small standalone chat-completion request with
// tools/tool_choice set. It does NOT use ToolManager's normal loop (which is
// disabled for 'quiet'-type generations anyway) and never touches the prompt.
//
// How it hooks in (extension-only, no core edits):
//  1. GENERATION_AFTER_COMMANDS fires before Generate() dispatches to
//     generateGroupWrapper(); it is awaited, so we can run the director probe here.
//  2. We mutate the group IN MEMORY: reorder members + mute non-selected + set
//     activation_strategy = LIST, so activateListOrder() produces exactly our roster.
//  3. generateGroupWrapper() has computed its activation by the time it emits
//     GROUP_WRAPPER_STARTED, so we restore the group data right there - the
//     actual member generations always run against pristine group state.
//  4. In chained mode we then probe again after every reply and drive further
//     beats via generateGroupWrapper(false, 'normal', { force_chid }).

import {
    addOneMessage,
    chat,
    characters,
    getRequestHeaders,
    name1,
    saveChatConditional,
    saveSettingsDebounced,
    substituteParams,
    system_avatar,
    systemUserName,
} from '../../../../script.js';
import { eventSource, event_types } from '../../../events.js';
import {
    editGroup,
    generateGroupWrapper,
    group_activation_strategy,
    groups,
    openGroupId,
    selected_group,
} from '../../../group-chats.js';
// Live binding: reading is allowed (writing is not possible/importable anyway).
// It is true while generateGroupWrapper() is running its member loop.
import { is_group_generating, is_group_automode_enabled } from '../../../group-chats.js';
import {
    chat_completion_sources,
    custom_prompt_post_processing_types,
    getChatCompletionModel,
    oai_settings,
} from '../../../openai.js';
import { ChatCompletionService } from '../../../custom-request.js';
import { delay } from '../../../utils.js';
import { extension_settings } from '../../../extensions.js';
import { SlashCommandParser } from '../../../slash-commands/SlashCommandParser.js';
import { SlashCommand } from '../../../slash-commands/SlashCommand.js';
import { ARGUMENT_TYPE, SlashCommandArgument } from '../../../slash-commands/SlashCommandArgument.js';
import { DOMPurify } from '../../../../lib.js';

// #region Constants

const SETTINGS_KEY = 'groupSceneDirector';
const TOOL_NAME = 'scene_director';
const END_TURN_SENTINEL = 'END_TURN';

/** Custom "Group reply strategy" option injected into the group edit panel. */
const SD_STRATEGY = 4;
/** Safety cap when the user enables unbounded recursion (0 = infinite). */
const ABSOLUTE_RECURSION_CAP = 50;
/** Retry limit when "Full recursive" forbids END_TURN but the model insists. */
const FULL_MODE_LAST_RESORT = 1;

const DIRECTOR_DEFAULT_PROMPT = `You are the scene director of a group roleplay. Your only job is to decide WHO speaks or acts next - never write dialogue yourself.

Roster (answer with an entry copied EXACTLY):
{{roster}}

Rules:
- Answer ONLY by calling the ${TOOL_NAME} tool. Never answer with plain text.
- "next_speaker" must be an exact roster entry from the list above, or "${END_TURN_SENTINEL}".
- "{{justSpoke}}" just finished speaking. Do NOT pick them again unless they must respond to something clearly addressed at them.
- Spread the spotlight: prefer members who have not spoken recently.
- If several members should react, pick the most urgent one - the scene direction will ask again afterwards.
- Keep the decision grounded in the recent conversation.`;

const RULE_FIRST = `This is the START of a turn: the user just sent a new message (marked with >>> below). Exactly one member should reply first. Do NOT return ${END_TURN_SENTINEL}; pick whoever reacts most naturally.`;

const RULE_CONTINUE = `A new reply was just added (marked <<< below). Decide whether the turn should continue:
- If a member would realistically speak, react, or act on it, pick them.
- If the scene reached its natural beat, return "${END_TURN_SENTINEL}".`;

const RULE_FULL = `FULL RECURSIVE MODE: this group must keep the scene going - choosing "${END_TURN_SENTINEL}" is FORBIDDEN right now. You MUST pick an eligible member. Endless repetition is preferable to ending early.`;

// #endregion

// #region Settings

function defaultSettings() {
    return {
        enabled: false,
        mode: 'chained', // 'chained' | 'batch'
        strategies: {
            natural: true,
            pool: true,
            list: false,
            manual: false,
        },
        maxSpeakersPerTurn: 5,
        allowRepeat: 'group', // 'group' | 'always' | 'never'
        toolChoice: 'forced',
        maxTokens: 300,
        temperature: 0.2,
        timeoutMs: 20000,
        retries: 1,
        fallbackOnFailure: true,
        historyDepth: 16,
        messageCharLimit: 400,
        memberInfo: 'short', // 'name' | 'short' | 'full'
        memberInfoCharLimit: 300,
        stopBetweenMs: 0,
        showDecisions: 'hidden', // 'hidden' | 'toast' | 'system'
        extraPrompt: '',
    };
}

function loadSettings() {
    extension_settings[SETTINGS_KEY] = Object.assign(defaultSettings(), extension_settings[SETTINGS_KEY] ?? {});
    return extension_settings[SETTINGS_KEY];
}

const settings = loadSettings();

function saveSettings() {
    saveSettingsDebounced();
}

// #endregion

// #region State

const state = {
    turn: null,         // active turn while we steer the roster
    lastDecision: null, // last director decision (for /director debug)
    lastPayload: null,  // last director request payload (for /director debug)
    lastError: null,
    lastTest: null,     // tool-calling self test result
    logLines: [],       // debug log ring buffer (rendered into the log textbox)
};

const LOG_MAX_LINES = 500;

function logDir(text, { level = 'info' } = {}) {
    const time = new Date().toLocaleTimeString();
    state.logLines.push(`[${time}] ${text}`);
    if (state.logLines.length > LOG_MAX_LINES) {
        state.logLines = state.logLines.slice(-LOG_MAX_LINES);
    }
    (level === 'error' ? console.error : level === 'warn' ? console.warn : console.log)('[SceneDirector]', text);
    renderLog();
}

let logRenderPending = false;
function renderLog() {
    if (logRenderPending) return;
    logRenderPending = true;
    setTimeout(() => {
        logRenderPending = false;
        const el = getElement('log');
        if (!el?.length) return;
        el.val(state.logLines.join('\n'));
        el.scrollTop(el[0].scrollHeight);
    }, 200);
}

function isBusy() {
    return state.turn !== null;
}

// #endregion

// #region Group helpers

function getGroup() {
    if (!selected_group) return null;
    return groups.find(x => x.id === selected_group) ?? null;
}

function enabledAvatars(group) {
    const members = Array.isArray(group?.members) ? group.members : [];
    const disabled = Array.isArray(group?.disabled_members) ? group.disabled_members : [];
    return members.filter(a => !disabled.includes(a));
}

function charNameByAvatar(avatar) {
    return characters.find(x => x.avatar === avatar)?.name ?? 'Unknown';
}

/** Roster entry handed to the model: "<avatar>: <name>". Avatars must not contain ': '. */
function rosterEntry(avatar) {
    return `${avatar}: ${charNameByAvatar(avatar)}`;
}

function parseRosterEntry(value, roster) {
    const text = String(value ?? '').trim();
    const idx = text.indexOf(': ');
    if (idx > 0 && roster.includes(text.slice(0, idx))) {
        return text.slice(0, idx);
    }
    if (roster.includes(text)) return text;
    const byName = roster.find(a => charNameByAvatar(a).toLowerCase() === text.toLowerCase());
    return byName ?? null;
}

function resolveAllowRepeat(group) {
    if (settings.allowRepeat === 'always') return true;
    if (settings.allowRepeat === 'never') return false;
    return Boolean(group.allow_self_responses);
}

function isDirectorStrategy(group) {
    return Number(group?.activation_strategy ?? group_activation_strategy.NATURAL) === SD_STRATEGY;
}

// #region Per-group director settings

/**
 * Seeds per-group director fields so the group edit panel has sane defaults.
 * Persisted together with the group object by /api/groups/edit.
 */
function ensureGroupDirectorDefaults(group) {
    if (!group) return null;
    if (typeof group.sd_recursive !== 'boolean') group.sd_recursive = settings.mode === 'chained';
    if (group.sd_recursion_limit === undefined || group.sd_recursion_limit === null) group.sd_recursion_limit = 5;
    if (typeof group.sd_full_recursive !== 'boolean') group.sd_full_recursive = false;
    return group;
}

function resolveTurnMode(group) {
    ensureGroupDirectorDefaults(group);
    if (typeof group.sd_recursive === 'boolean') {
        return group.sd_recursive ? 'chained' : 'batch';
    }
    return settings.mode;
}

/**
 * Hard cap of AI replies for a chained/recursive turn.
 * A per-group limit of 0 means "infinite" and is clamped to ABSOLUTE_RECURSION_CAP.
 */
function resolveChainCap(group) {
    const raw = Number(group.sd_recursion_limit ?? 5);
    const clamped = Math.min(Math.max(1, isNaN(raw) ? 5 : raw), ABSOLUTE_RECURSION_CAP);
    if (raw === 0) {
        logDir(`Group "${group.name ?? group.id}" requested infinite recursion - hard-capped at ${ABSOLUTE_RECURSION_CAP} for safety`);
        return ABSOLUTE_RECURSION_CAP;
    }
    return clamped;
}

function strategyApplies(group) {
    // Explicit "Scene Director" strategy: always director-driven
    // (the master switch below still applies).
    if (isDirectorStrategy(group)) return true;
    const strategy = Number(group?.activation_strategy ?? group_activation_strategy.NATURAL);
    const map = {
        [group_activation_strategy.NATURAL]: 'natural',
        [group_activation_strategy.POOLED]: 'pool',
        [group_activation_strategy.LIST]: 'list',
        [group_activation_strategy.MANUAL]: 'manual',
    };
    return Boolean(map[strategy] && settings.strategies[map[strategy]]);
}

// #endregion

function lastNonUserMessageAvatar() {
    for (let i = chat.length - 1; i >= 0; i--) {
        const m = chat[i];
        if (m.is_user || m.is_system) continue;
        return m.original_avatar ?? null;
    }
    return null;
}

// #endregion

// #region Prompt building

function formatMemberBlurb(character) {
    if (!character || settings.memberInfo === 'name') return '';
    const body = String(character.description ?? '').replace(/\s+/g, ' ').trim();
    if (!body || settings.memberInfo === 'full') return body ? ` - ${body}` : '';
    const limit = Math.max(0, Number(settings.memberInfoCharLimit) || 300);
    const text = limit > 0 && body.length > limit ? `${body.slice(0, limit)}...` : body;
    return text ? ` - ${text}` : '';
}

function buildRosterText(group) {
    return enabledAvatars(group)
        .map((a, i) => `${i + 1}. \`${rosterEntry(a)}\`${formatMemberBlurb(characters.find(x => x.avatar === a))}`)
        .join('\n');
}

function buildRecentChatText() {
    const lines = [];
    const recent = chat.slice(-Math.max(1, Number(settings.historyDepth) || 16));
    for (const message of recent) {
        if (message.is_system) continue;
        const speaker = message.is_user ? name1 : (message.name || 'Unknown');
        const limit = Math.max(50, Number(settings.messageCharLimit) || 400);
        let body = String(message.mes ?? '').trim();
        if (!body) continue;
        if (body.length > limit) body = `${body.slice(0, limit)}...`;
        lines.push(`${speaker}: ${body}`);
    }
    return lines.join('\n\n');
}

function buildDirectorMessages(group, { stage, justSpoke, latestUserInput, mustContinue }) {
    const allowRepeat = resolveAllowRepeat(group);
    const base = substituteParams(
        settings.extraPrompt
            ? `${DIRECTOR_DEFAULT_PROMPT}\n\nCustom instructions:\n${settings.extraPrompt}`
            : DIRECTOR_DEFAULT_PROMPT)
        .replaceAll('{{roster}}', buildRosterText(group))
        .replaceAll('{{justSpoke}}', justSpoke ? charNameByAvatar(justSpoke) : '(nobody yet)')
        .replaceAll('{{latestUserMessage}}', latestUserInput ?? '');

    const chatText = buildRecentChatText();
    let note;
    if (stage === 'first') {
        note = latestUserInput
            ? `${chatText}\n\n>>> (NEW user message from ${name1}):\n${name1}: ${latestUserInput}`
            : chatText;
    } else {
        note = `${chatText}\n\n<<< (the latest reply above was just added)`;
    }

    const rules = [stage === 'first' ? RULE_FIRST : RULE_CONTINUE];
    if (mustContinue) rules.push(RULE_FULL);

    const system = [
        base,
        ...rules,
        `Immediate previous speaker may speak again: ${allowRepeat ? 'yes' : 'no'}.`,
        '',
        note,
    ].join('\n');

    return [
        { role: 'system', content: system },
        { role: 'user', content: 'Decide who goes next.' },
    ];
}

function buildDirectorTool(group) {
    const roster = enabledAvatars(group).map(rosterEntry);
    return {
        type: 'function',
        function: {
            name: TOOL_NAME,
            description: 'Pick the character who speaks or acts next in the group scene, or end the turn.',
            parameters: {
                type: 'object',
                properties: {
                    next_speaker: {
                        type: 'string',
                        enum: [END_TURN_SENTINEL, ...roster],
                        description: `Exact roster entry of the character who should speak next, or "${END_TURN_SENTINEL}" if nobody should.`,
                    },
                    reasoning: {
                        type: 'string',
                        description: 'Short reason (1-2 sentences) for this decision.',
                    },
                },
                required: ['next_speaker', 'reasoning'],
            },
        },
    };
}

// #endregion

// #region Director request

async function withTimeout(run, signal = null) {
    const controller = new AbortController();
    const onAbort = () => controller.abort(signal?.reason ?? new Error('Scene Director aborted'));
    if (signal) {
        if (signal.aborted) onAbort();
        else signal.addEventListener('abort', onAbort, { once: true });
    }
    const timeoutMs = Math.max(500, Number(settings.timeoutMs) || 20000);
    const timer = setTimeout(
        () => controller.abort(new DOMException('Director request timed out', 'TimeoutError')),
        timeoutMs,
    );
    try {
        return await run(controller.signal);
    } finally {
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
    }
}

/**
 * Runs one director probe. Returns a decision:
 * { shouldContinue: boolean, avatar: string|null, reasoning: string }
 * or null when the request could not be completed.
 */
async function runDirectorProbe(group, signal, { stage, justSpoke, latestUserInput, mustContinue = false }, retriesLeft) {
    retriesLeft = Math.max(0, Number(retriesLeft) ?? 0);

    const payload = {
        stream: false,
        messages: buildDirectorMessages(group, { stage, justSpoke, latestUserInput, mustContinue }),
        model: getChatCompletionModel(oai_settings),
        chat_completion_source: oai_settings.chat_completion_source,
        max_tokens: Math.max(64, Number(settings.maxTokens) || 300),
        temperature: Math.min(1, Math.max(0, Number(settings.temperature) ?? 0.2)),
        custom_prompt_post_processing: custom_prompt_post_processing_types.NONE,
        tools: [buildDirectorTool(group)],
        tool_choice: settings.toolChoice === 'forced'
            ? { type: 'function', function: { name: TOOL_NAME } }
            : 'auto',
        include_reasoning: false,
    };
    if (oai_settings.chat_completion_source === chat_completion_sources.CUSTOM) {
        payload.custom_url = oai_settings.custom_url;
        if (!payload.model) {
            state.lastError = 'Scene Director needs a model name set for the Custom API';
            logDir(state.lastError, { level: 'error' });
            return null;
        }
    }

    state.lastPayload = payload;
    logDir(`Probe [${stage}] -> ${oai_settings.chat_completion_source} / ${payload.model} (tool_choice=${settings.toolChoice}${mustContinue ? ', full-recursive' : ''})`);

    let json;
    try {
        json = await withTimeout(
            probeSignal => ChatCompletionService.processRequest(payload, {}, false, probeSignal),
            signal,
        );
    } catch (error) {
        state.lastError = `Director request failed: ${error?.message ?? error}`;
        logDir(state.lastError, { level: 'error' });
        return null;
    }

    const choices = Array.isArray(json?.choices) ? json.choices : [];
    const choice = choices.find(c => c?.index === 0) ?? choices[0] ?? null;
    const toolCalls = Array.isArray(choice?.message?.tool_calls) ? choice.message.tool_calls : [];

    // Log the model's raw output for the debug textbox
    if (choice?.message?.content) {
        logDir(`Model output: ${String(choice.message.content).slice(0, 400)}`);
    }
    if (toolCalls.length) {
        try {
            logDir(`Tool call(s): ${JSON.stringify(toolCalls.map(t => ({ name: t?.function?.name, arguments: t?.function?.arguments }))).slice(0, 700)}`);
        } catch { /* ignore serialization issues */ }
    }

    const toolCall = toolCalls.find(t => t?.function?.name === TOOL_NAME) ?? null;

    if (!toolCall) {
        state.lastError = 'Director did not return a tool call';
        if (retriesLeft > 0) {
            logDir(`${state.lastError} - retrying (${retriesLeft} left)`, { level: 'warn' });
            await delay(250);
            return runDirectorProbe(group, signal, { stage, justSpoke, latestUserInput, mustContinue }, retriesLeft - 1);
        }
        logDir(state.lastError, { level: 'warn' });
        return null;
    }

    let args = null;
    try {
        args = JSON.parse(String(toolCall.function.arguments));
    } catch {
        args = { next_speaker: toolCall.function.arguments };
    }

    const roster = enabledAvatars(group);
    const rawSpeaker = String(args?.next_speaker ?? '').trim() || END_TURN_SENTINEL;
    const avatar = rawSpeaker === END_TURN_SENTINEL ? null : parseRosterEntry(rawSpeaker, roster);

    // Unknown member: retry with a notice
    if (avatar === null && rawSpeaker !== END_TURN_SENTINEL && retriesLeft > 0) {
        state.lastError = `Director picked an unknown member: ${rawSpeaker}`;
        logDir(state.lastError, { level: 'warn' });
        await delay(250);
        return runDirectorProbe(group, signal, { stage, justSpoke, latestUserInput, mustContinue }, retriesLeft - 1);
    }

    // Repeat policy: the immediate previous speaker may be banned from going again
    if (avatar && avatar === justSpoke && !resolveAllowRepeat(group) && retriesLeft > 0) {
        state.lastError = `Director re-picked the previous speaker (${charNameByAvatar(avatar)}) while repeats are disallowed`;
        logDir(state.lastError, { level: 'warn' });
        await delay(250);
        return runDirectorProbe(group, signal, { stage, justSpoke, latestUserInput, mustContinue }, retriesLeft - 1);
    }

    // Full recursive mode: END_TURN is forbidden while the target is unmet
    if (mustContinue && avatar === null) {
        if (retriesLeft > 0) {
            state.lastError = 'Director chose END_TURN while Full Recursive mode is active - retrying';
            logDir(state.lastError, { level: 'warn' });
            await delay(250);
            return runDirectorProbe(group, signal, { stage, justSpoke, latestUserInput, mustContinue }, retriesLeft - 1);
        }
        logDir(`Director keeps refusing to continue in Full Recursive mode - giving up after ${FULL_MODE_LAST_RESORT} attempts`, { level: 'warn' });
    }

    const decision = {
        shouldContinue: avatar !== null,
        avatar,
        reasoning: String(args?.reasoning ?? '').trim(),
        stage,
        raw: rawSpeaker,
        mustContinue,
    };
    state.lastDecision = decision;
    state.lastError = null;
    announceDecision(decision);
    return decision;
}

// #endregion

// #region Decision feedback

function announceDecision(decision) {
    const who = decision.avatar ? charNameByAvatar(decision.avatar) : 'Nobody (end turn)';
    const label = `Scene: ${who}${decision.reasoning ? ` - ${decision.reasoning}` : ''}`;
    switch (settings.showDecisions) {
        case 'toast':
            toastr.info(DOMPurify.sanitize(label), 'Scene Director', { timeOut: 4000 });
            break;
        case 'system':
            void saveDecisionMessage(label);
            break;
        default:
            console.log('[SceneDirector]', label);
            break;
    }
}

async function saveDecisionMessage(text) {
    const message = {
        name: systemUserName,
        force_avatar: system_avatar,
        is_system: true,
        is_user: false,
        mes: text,
        extra: { isSmallSys: true, type: 'scene_director_decision' },
    };
    chat.push(message);
    addOneMessage(message);
    await saveChatConditional();
}

// #endregion

// #region Group snapshot / apply / restore

function snapshotGroup(group) {
    return {
        members: [...(group?.members ?? [])],
        disabled: [...(group?.disabled_members ?? [])],
        strategy: Number(group?.activation_strategy ?? group_activation_strategy.NATURAL),
    };
}

function replaceInPlace(target, items) {
    if (!Array.isArray(target)) return;
    target.length = 0;
    target.push(...items);
}

/**
 * Rewrites the group IN MEMORY so generateGroupWrapper()'s LIST activation
 * produces exactly `orderedAvatars` in order: selected first + everyone else
 * temporarily muted. Call restoreGroup() with the returned snapshot afterwards.
 * NOTE: in APPEND groups this temporarily excludes other members' merged cards
 * in the first generation (getGroupCharacterCards skips disabled members).
 */
function applyRoster(group, orderedAvatars) {
    const snapshot = snapshotGroup(group);
    if (!Array.isArray(group.disabled_members)) group.disabled_members = [];
    const others = enabledAvatars(group).filter(a => !orderedAvatars.includes(a));
    group.activation_strategy = group_activation_strategy.LIST;
    replaceInPlace(group.members, [...orderedAvatars]);
    replaceInPlace(group.disabled_members, [...group.disabled_members, ...others]);
    return snapshot;
}

/** Mutes everyone: the turn will have zero AI replies (used when fallback is disabled). */
function applyEmptyTurn(group) {
    const snapshot = snapshotGroup(group);
    if (!Array.isArray(group.disabled_members)) group.disabled_members = [];
    group.activation_strategy = group_activation_strategy.LIST;
    replaceInPlace(group.disabled_members, [...enabledAvatars(group)]);
    return snapshot;
}

function restoreGroup(group, snapshot) {
    if (!group || !snapshot) return;
    group.activation_strategy = snapshot.strategy;
    if (Array.isArray(group.members)) replaceInPlace(group.members, snapshot.members);
    if (!Array.isArray(group.disabled_members)) group.disabled_members = [];
    replaceInPlace(group.disabled_members, snapshot.disabled);
}

// #endregion

// #region Turn flow

/**
 * Runs the director and prepares the roster for the original wrapper.
 * Runs during the awaited GENERATION_AFTER_COMMANDS event for the user's turn.
 */
async function onGenerationAfterCommands(type, options = {}, dryRun = false) {
    try {
        if (dryRun || !settings.enabled) return;
        if (type !== 'normal') return; // swipes/continues/impersonate stay stock
        if (isBusy() || is_group_generating) return;
        if (is_group_automode_enabled) return; // automode runs its own wrapper directly

        const group = getGroup();
        if (!group || !Array.isArray(group.members) || !group.members.length) return;
        if (!strategyApplies(group)) return;

        const roster = enabledAvatars(group);
        if (!roster.length) return;

        const signal = options?.signal instanceof AbortSignal ? options.signal : new AbortController().signal;
        const latestUserInput = getLatestUserInput();
        const mode = resolveTurnMode(group);
        const fullRecursive = Boolean(group.sd_full_recursive ?? false);
        const chainCap = resolveChainCap(group);

        state.turn = {
            groupId: group.id,
            mode,
            phase: mode === 'chained' ? 'first' : 'batch',
            snapshot: null,
            signal,
            lastSpeaker: null,
            count: 0,
            chainCap,
            full: fullRecursive,
        };
        logDir(`Turn start: mode=${mode}, cap=${chainCap}, full-recursive=${fullRecursive}, strategy=${group.activation_strategy}`);

        const justSpoke = lastNonUserMessageAvatar();
        const first = await runDirectorProbe(group, signal, { stage: 'first', justSpoke, latestUserInput, mustContinue: false }, settings.retries);

        if (!first?.avatar) {
            state.lastError = state.lastError ?? 'Director did not pick a first speaker';
            toastr.warning(state.lastError, 'Scene Director');
            logDir(`First speaker: none (fallback=${settings.fallbackOnFailure ? 'stock activation' : 'silent turn'})`, { level: 'warn' });
            if (settings.fallbackOnFailure) {
                restoreOnAbandon(); // stock activation proceeds
            } else {
                state.turn.snapshot = applyEmptyTurn(group); // nobody speaks this turn
                state.turn.phase = 'silent';
            }
            return;
        }
        logDir(`First speaker: ${charNameByAvatar(first.avatar)}`);

        let rosterToApply;
        if (mode === 'batch') {
            rosterToApply = await pickBatchRoster(group, signal, justSpoke, latestUserInput, first.avatar, chainCap, fullRecursive);
            rosterToApply = rosterToApply.filter(onlyUnique);
        } else {
            rosterToApply = [first.avatar];
            state.turn.lastSpeaker = first.avatar;
        }

        state.turn.snapshot = applyRoster(group, rosterToApply);
        state.turn.count = rosterToApply.length;
        logDir(`Roster applied: ${rosterToApply.map(a => charNameByAvatar(a)).join(' -> ')} (group data will be restored on wrapper start)`);
        // The original Generate() now flows into generateGroupWrapper(), which
        // activates exactly rosterToApply in order. Continuation (chained mode)
        // is picked up in onGroupWrapperFinished().
    } catch (error) {
        console.error('[SceneDirector] Turn setup failed:', error);
        restoreOnAbandon();
    }
}

function onlyUnique(value, index, array) {
    return array.indexOf(value) === index;
}

function getLatestUserInput() {
    // At GENERATION_AFTER_COMMANDS time the user's message is NOT yet in `chat`
    // for group turns - Generate adds it inside the first member round
    // (textarea is consumed at script.js ~4398-4453). The textarea still holds
    // the pending text at this point, so prefer it.
    const pending = String($('#send_textarea').val() ?? '').trim();
    if (pending) return pending;
    const last = chat[chat.length - 1];
    return last?.is_user ? String(last.mes) : null;
}

/** Batch mode: queue additional speakers beyond the first one. */
async function pickBatchRoster(group, signal, justSpoke, latestUserInput, firstAvatar, chainCap, fullRecursive) {
    const globalBatchMax = Math.max(1, Number(settings.maxSpeakersPerTurn));
    const target = fullRecursive && chainCap ? Math.min(chainCap, globalBatchMax) : globalBatchMax;
    const roster = [firstAvatar];
    let spoken = justSpoke;
    let mustContinue = fullRecursive && target > roster.length;
    while (roster.length < target) {
        const decision = await runDirectorProbe(group, signal, { stage: 'continue', justSpoke: spoken, latestUserInput, mustContinue }, 0);
        if (!decision?.shouldContinue) break;
        roster.push(decision.avatar);
        spoken = decision.avatar;
        mustContinue = fullRecursive && target > roster.length;
    }
    const capping = fullRecursive && chainCap > target ? chainCap : 0;
    if (capping) logDir(`Batch full-recursive target ${chainCap} clamped by batch setting ${globalBatchMax}`);
    return roster;
}

function restoreOnAbandon() {
    const turn = state.turn;
    if (turn?.snapshot) {
        restoreGroup(groups.find(x => x.id === turn.groupId), turn.snapshot);
    }
    state.turn = null;
}

/** Runs the chained (recursive) continuation loop until the cap or a natural end. */
async function runChainedTurn(turn) {
    try {
        const group = groups.find(x => x.id === turn.groupId);
        if (!group) return;

        while (turn.count < turn.chainCap) {
            // Full recursive: forbid END_TURN while the turn should keep going
            const mustContinue = Boolean(turn.full) && turn.count < turn.chainCap;
            const decision = await runDirectorProbe(
                group, turn.signal,
                { stage: 'continue', justSpoke: turn.lastSpeaker, latestUserInput: null, mustContinue },
                settings.retries,
            );
            if (!decision || !decision.shouldContinue) break;

            const chId = characters.findIndex(y => y.avatar === decision.avatar);
            if (chId === -1) break;

            if (Number(settings.stopBetweenMs) > 0) await delay(Number(settings.stopBetweenMs));
            if (turn.signal?.aborted) break;

            turn.count++;
            logDir(`Recursive beat ${turn.count}: ${charNameByAvatar(decision.avatar)} speaking`);
            await generateGroupWrapper(false, 'normal', {
                force_chid: chId,
                signal: turn.signal,
            });
            turn.lastSpeaker = decision.avatar;
        }
        logDir(`Chain ended after ${turn.count} spoken beat(s)${turn.signal?.aborted ? ' (aborted)' : ''}`);
    } catch (error) {
        if (error?.name !== 'AbortError') {
            console.error('[SceneDirector] Chain interrupted:', error);
            logDir(`Chain interrupted: ${error?.message ?? error}`, { level: 'error' });
        }
    } finally {
        restoreGroup(groups.find(x => x.id === turn.groupId), turn.snapshot);
        state.turn = null;
    }
}

/**
 * Fires when generateGroupWrapper() starts its member loop. At this point the
 * wrapper has ALREADY computed its activated roster from our mutated group,
 * so we can safely restore the original data immediately - all prompt building
 * (cards, depth prompts) then runs against pristine group state.
 */
async function onGroupWrapperStarted(payload = {}) {
    const turn = state.turn;
    if (!turn || (payload?.selected_group && payload.selected_group !== turn.groupId)) return;
    if (turn.phase === 'chain') return; // our own continuation rounds (force_chid)
    const group = groups.find(x => x.id === turn.groupId);
    restoreGroup(group, turn.snapshot);
    turn.snapshot = null;
}

/**
 * Fires when a generateGroupWrapper() pass finishes.
 * - Safety net restore (e.g. aborted before GROUP_WRAPPER_STARTED).
 * - Chained mode seam: after the first (roster-of-one) round, start the
 *   continuation loop. Awaited by the wrapper, so the send button stays busy.
 */
async function onGroupWrapperFinished(payload = {}) {
    const turn = state.turn;
    if (!turn || (payload?.selected_group && payload.selected_group !== turn.groupId)) return;

    if (turn.signal?.aborted) {
        restoreGroup(groups.find(x => x.id === turn.groupId), turn.snapshot);
        state.turn = null;
        return;
    }

    if (turn.snapshot) {
        // Aborted before STARTED, or the wrapper exited without emitting it
        restoreGroup(groups.find(x => x.id === turn.groupId), turn.snapshot);
        turn.snapshot = null;
    }

    if (turn.phase === 'first') {
        turn.phase = 'chain';
        await runChainedTurn(turn);
        return;
    }

    if (turn.phase !== 'chain') {
        // batch or silent turns end with the original wrapper
        state.turn = null;
    }
    // phase 'chain': runChainedTurn() owns this turn and cleans up
}

// #endregion (Turn flow)

// #region Group edit panel injection

const GROUP_OPTION_HTML = `<option value="4">Scene Director (recursive)</option>`;

const GROUP_BLOCK_HTML = `
<label class="checkbox_label whitespacenowrap" id="sceneDirector_group_recursive_label" title="Let the Scene Director extension carry the scene: it probes the chat for suitable speakers via tool calls.">
    <input id="sceneDirector_group_recursive" type="checkbox" />
    <span>Scene Director: Recursive</span>
    <input id="sceneDirector_group_recursion_limit" class="text_pole textarea_compact widthUnset" type="number" min="0" max="50" step="1" value="5" title="Max AI replies per recursive scene turn. 0 = run until the director ends the scene (capped at ${ABSOLUTE_RECURSION_CAP} for safety)." />
</label>
<label class="checkbox_label whitespacenowrap" id="sceneDirector_group_full_label" title="Full recursive: the director MUST reach the recursion limit each turn (0 above = run forever, stop button always works).">
    <input id="sceneDirector_group_full_recursive" type="checkbox" />
    <span>Full recursive</span>
</label>`;

function injectGroupEditControls() {
    // New custom "Group reply strategy" option
    const strategySelect = $('#rm_group_activation_strategy');
    if (strategySelect.length && !strategySelect.find(`option[value="${SD_STRATEGY}"]`).length) {
        strategySelect.append(GROUP_OPTION_HTML);
    }

    // Recursive controls next to "Allow self responses" / "Auto Mode"
    const anchor = $('#rm_group_hidemutedsprites_label');
    if (anchor.length && !$('#sceneDirector_group_recursive').length) {
        const block = $(GROUP_BLOCK_HTML);
        anchor.parent().append(block);
        block.find('#sceneDirector_group_recursive').on('change', async function () {
            const group = ensureGroupDirectorDefaults(openGroupId && groups.find(x => x.id === openGroupId));
            if (!group) return;
            group.sd_recursive = $(this).prop('checked');
            logDir(`Group "${group.name}": recursive = ${group.sd_recursive}`);
            await saveGroupFields(group);
        });
        block.find('#sceneDirector_group_full_recursive').on('change', async function () {
            const group = ensureGroupDirectorDefaults(openGroupId && groups.find(x => x.id === openGroupId));
            if (!group) return;
            group.sd_full_recursive = $(this).prop('checked');
            logDir(`Group "${group.name}": full recursive = ${group.sd_full_recursive}`);
            await saveGroupFields(group);
        });
        block.find('#sceneDirector_group_recursion_limit').on('change', async function () {
            const group = ensureGroupDirectorDefaults(openGroupId && groups.find(x => x.id === openGroupId));
            if (!group) return;
            const raw = Number(String($(this).val() ?? ''));
            group.sd_recursion_limit = Math.min(Math.max(0, isNaN(raw) ? 0 : Math.floor(raw)), ABSOLUTE_RECURSION_CAP);
            $(this).val(group.sd_recursion_limit);
            logDir(`Group "${group.name}": recursion limit = ${group.sd_recursion_limit === 0 ? 'infinite (capped ' + ABSOLUTE_RECURSION_CAP + ')' : group.sd_recursion_limit}`);
            await saveGroupFields(group);
        });
    }
}

/** Persists per-group fields through the same core path the group panel uses. */
async function saveGroupFields(group) {
    try {
        await editGroup(group.id, false, false);
    } catch (error) {
        logDir(`Failed to save group settings: ${error?.message ?? error}`, { level: 'error' });
    }
}

/** Mirrors per-group state into the injected panel inputs. */
function syncGroupEditControls() {
    const group = ensureGroupDirectorDefaults(openGroupId && groups.find(x => x.id === openGroupId));
    $('#sceneDirector_group_recursive').prop('checked', Boolean(group?.sd_recursive));
    $('#sceneDirector_group_full_recursive').prop('checked', Boolean(group?.sd_full_recursive));
    $('#sceneDirector_group_recursion_limit').val(group?.sd_recursion_limit ?? 5);
}

// #endregion

// #region Settings UI (Extension drawer) - includes Test & Debug section

/** Sends a minimal probe to verify the current API/model actually works with tools. */
async function runToolingSelfTest() {
    const started = Date.now();
    logDir('--- Tool calling self test: start ---');
    const button = getElement('testTool');
    try {
        button?.prop('disabled', true);
        const roster = getGroup() ? enabledAvatars(getGroup()) : [];
        const tool = buildDirectorTool({ members: [], disabled_members: [] });
        const payload = {
            stream: false,
            messages: [
                { role: 'system', content: 'This is a connectivity and function-calling self test for the Group Scene Director extension.' },
                { role: 'user', content: `Call the ${TOOL_NAME} tool exactly once: set next_speaker to "${END_TURN_SENTINEL}" and put a short confirmation sentence into reasoning. Do not write plain text.` },
            ],
            model: getChatCompletionModel(oai_settings),
            chat_completion_source: oai_settings.chat_completion_source,
            max_tokens: Math.max(64, Number(settings.maxTokens) || 300),
            temperature: 0,
            custom_prompt_post_processing: custom_prompt_post_processing_types.NONE,
            tools: [tool],
            tool_choice: settings.toolChoice === 'forced'
                ? { type: 'function', function: { name: TOOL_NAME } }
                : 'auto',
            include_reasoning: false,
        };
        if (roster.length) logDir(`Testing with a roster context of ${roster.length} member(s)`);
        logDir(`Test request -> ${oai_settings.chat_completion_source} / ${payload.model || '(no model set!)'}`);

        if (oai_settings.chat_completion_source === chat_completion_sources.CUSTOM) {
            payload.custom_url = oai_settings.custom_url;
            if (!payload.model) {
                throw new Error('No model name set for the Custom API - set it in the OpenAI API panel first.');
            }
        }

        const json = await withTimeout(signal => ChatCompletionService.processRequest(payload, {}, false, signal), null);
        const choices = Array.isArray(json?.choices) ? json.choices : [];
        const choice = choices.find(c => c?.index === 0) ?? choices[0] ?? null;
        const toolCalls = Array.isArray(choice?.message?.tool_calls) ? choice.message.tool_calls : [];

        if (choice?.message?.content) {
            logDir(`Model text output: ${String(choice.message.content).slice(0, 400)}`);
        }
        for (const toolCall of toolCalls) {
            logDir(`Tool call received: ${toolCall?.function?.name}(${String(toolCall?.function?.arguments ?? '').slice(0, 500)})`);
        }

        const call = toolCalls.find(t => t?.function?.name === TOOL_NAME) ?? null;
        if (call) {
            let args = null;
            try { args = JSON.parse(String(call.function.arguments)); } catch { args = { next_speaker: call.function.arguments }; }
            state.lastTest = { ok: true, args, ms: Date.now() - started };
            logDir(`SELF TEST PASSED in ${Date.now() - started} ms - tool calling works for ${oai_settings.chat_completion_source}`);
            toastr.success('Tool calling works - the Scene Director can use this API/model.', 'Scene Director');
        } else {
            state.lastTest = { ok: false, error: 'No scene_director tool call in response', ms: Date.now() - started };
            logDir('SELF TEST FAILED: the response contained no scene_director tool call. Either unsupported tool API or tool_choice was ignored by the backend.', { level: 'error' });
            toastr.error('Tool calling self test failed - open the log for details.', 'Scene Director');
        }
    } catch (error) {
        state.lastTest = { ok: false, error: String(error?.message ?? error), ms: Date.now() - started };
        logDir(`SELF TEST FAILED: ${error?.message ?? error}`, { level: 'error' });
        toastr.error(String(error?.message ?? error), 'Scene Director');
    } finally {
        button?.prop('disabled', false);
        logDir(`--- Tool calling self test: end (${Date.now() - started} ms) ---`);
    }
}

const SETTINGS_HTML = `
<div id="sceneDirector" class="scene-director extension_settings">
    <div class="inline-drawer">
        <div class="inline-drawer-toggle inline-drawer-header">
            <b>Group Scene Director</b>
            <div class="inline-drawer-icon fa-solid fa-circle-down down"></div>
        </div>
        <div class="inline-drawer-content">
            <label class="checkbox_label">
                <input id="sceneDirector_enabled" type="checkbox">
                <span>Enable director-driven group selection <small>(LLM picks speakers via tool calls)</small></span>
            </label>

            <label for="sceneDirector_mode">Selection mode</label>
            <select id="sceneDirector_mode" class="text_pole">
                <option value="chained">Chained - decide after every reply (recursive scene)</option>
                <option value="batch">Batch - queue the whole turn up front</option>
            </select>

            <div>
                <b>May override activation strategies</b>
                <div class="scene-director-strategies">
                    <label><input id="sceneDirector_strategies_natural" type="checkbox"><span>Natural</span></label>
                    <label><input id="sceneDirector_strategies_pool" type="checkbox"><span>Pool</span></label>
                    <label><input id="sceneDirector_strategies_list" type="checkbox"><span>List order</span></label>
                    <label><input id="sceneDirector_strategies_manual" type="checkbox"><span>Manual</span></label>
                </div>
            </div>

            <label for="sceneDirector_allowRepeat">Same character may speak twice</label>
            <select id="sceneDirector_allowRepeat" class="text_pole">
                <option value="group">Follow group "self-responses" setting</option>
                <option value="always">Always allowed</option>
                <option value="never">Never allowed</option>
            </select>

            <label for="sceneDirector_maxSpeakersPerTurn">Batch roster size (chained turns use the per-group recursion limit): <span id="sceneDirector_maxSpeakersPerTurn_out"></span></label>
            <input id="sceneDirector_maxSpeakersPerTurn" type="range" min="1" max="12" step="1">

            <label for="sceneDirector_showDecisions">Show decisions</label>
            <select id="sceneDirector_showDecisions" class="text_pole">
                <option value="hidden">Hidden (console only)</option>
                <option value="toast">Toast popup</option>
                <option value="system">Small system message in chat</option>
            </select>

            <details>
                <summary>Director request</summary>
                <label for="sceneDirector_toolChoice">Tool choice</label>
                <select id="sceneDirector_toolChoice" class="text_pole">
                    <option value="forced">Forced (model must call the tool)</option>
                    <option value="auto">Auto</option>
                </select>

                <label for="sceneDirector_temperature">Temperature</label>
                <input id="sceneDirector_temperature" type="number" step="0.1" min="0" max="1">

                <label for="sceneDirector_maxTokens">Max tokens</label>
                <input id="sceneDirector_maxTokens" type="number" step="16" min="64">

                <label for="sceneDirector_timeoutMs">Timeout (ms)</label>
                <input id="sceneDirector_timeoutMs" type="number" step="500" min="500">

                <label for="sceneDirector_retries">Retries on invalid answer</label>
                <input id="sceneDirector_retries" type="number" step="1" min="0" max="5">

                <label class="checkbox_label">
                    <input id="sceneDirector_fallbackOnFailure" type="checkbox">
                    <span>Fall back to stock activation when the director fails</span>
                </label>
            </details>

            <details>
                <summary>Context &amp; members</summary>
                <label for="sceneDirector_historyDepth">Recent messages sent to director: <span id="sceneDirector_historyDepth_out"></span></label>
                <input id="sceneDirector_historyDepth" type="range" min="1" max="40" step="1">

                <label for="sceneDirector_messageCharLimit">Message character limit</label>
                <input id="sceneDirector_messageCharLimit" type="number" step="50" min="50">

                <label for="sceneDirector_memberInfo">Member info in roster</label>
                <select id="sceneDirector_memberInfo" class="text_pole">
                    <option value="name">Names only</option>
                    <option value="short">Short (first N chars of description)</option>
                    <option value="full">Full description (token heavy)</option>
                </select>

                <label for="sceneDirector_memberInfoCharLimit">Blurb character limit</label>
                <input id="sceneDirector_memberInfoCharLimit" type="number" step="50" min="0">
            </details>

            <details>
                <summary>Advanced</summary>
                <label for="sceneDirector_stopBetweenMs">Delay between chained replies (ms)</label>
                <input id="sceneDirector_stopBetweenMs" type="number" step="100" min="0">

                <label for="sceneDirector_extraPrompt">Extra director instructions (macros allowed, {{roster}}/{{justSpoke}} available)</label>
                <textarea id="sceneDirector_extraPrompt" class="text_pole textarea_compact" rows="3" placeholder="e.g. Prefer answers directed at {{user}}; keep debates spicy..."></textarea>
            </details>

            <details open>
                <summary>Test &amp; Debug</summary>
                <button id="sceneDirector_testTool" class="menu_button">Test tool calling</button>
                <label for="sceneDirector_log">Live log (requests, model output, errors)</label>
                <textarea id="sceneDirector_log" class="text_pole scene-director-log" rows="10" readonly spellcheck="false" wrap="off"></textarea>
                <button id="sceneDirector_clearLog" class="menu_button">Clear log</button>
            </details>
        </div>
    </div>
</div>`;

function getElement(id) {
    // Every control in this panel uses the "#sceneDirector_<id>" convention
    // (the "#sceneDirector" wrapper is only a container, not an id namespace).
    return $('#sceneDirector_' + id);
}

function applyOutLabels() {
    getElement('maxSpeakersPerTurn_out').text(settings.maxSpeakersPerTurn);
    getElement('historyDepth_out').text(settings.historyDepth);
}

function bindElements() {
    const bindControl = (id, apply, read) => {
        const el = getElement(id);
        // A missing control used to fail silently (empty jQuery set) and leave
        // the whole panel inert - report it instead.
        if (!el.length) {
            logDir(`Settings control not found: #sceneDirector_${id}`, { level: 'error' });
            return;
        }
        apply(el);
        el.on('change input', () => {
            read(el);
            applyOutLabels();
            saveSettings();
        });
    };

    const bindCheckboxValue = (id) => Boolean(getElement(id).prop('checked'));
    const bindSelectValue = (id) => String(getElement(id).val());

    // top-level values
    bindControl('enabled', el => el.prop('checked', settings.enabled), el => settings.enabled = el.prop('checked'));
    bindControl('mode', el => el.val(settings.mode), el => settings.mode = bindSelectValue('mode'));
    bindControl('allowRepeat', el => el.val(settings.allowRepeat), el => settings.allowRepeat = bindSelectValue('allowRepeat'));
    bindControl('maxSpeakersPerTurn', el => el.val(settings.maxSpeakersPerTurn), el => settings.maxSpeakersPerTurn = Number(el.val()) || 1);
    bindControl('showDecisions', el => el.val(settings.showDecisions), el => settings.showDecisions = bindSelectValue('showDecisions'));
    bindControl('toolChoice', el => el.val(settings.toolChoice), el => settings.toolChoice = bindSelectValue('toolChoice'));
    bindControl('temperature', el => el.val(settings.temperature), el => settings.temperature = Math.min(1, Math.max(0, Number(el.val()) || 0)));
    bindControl('maxTokens', el => el.val(settings.maxTokens), el => settings.maxTokens = Math.max(64, Number(el.val()) || 300));
    bindControl('timeoutMs', el => el.val(settings.timeoutMs), el => settings.timeoutMs = Math.max(500, Number(el.val()) || 20000));
    bindControl('retries', el => el.val(settings.retries), el => settings.retries = Math.min(5, Math.max(0, Number(el.val()) || 0)));
    bindControl('fallbackOnFailure', el => el.prop('checked', settings.fallbackOnFailure), el => settings.fallbackOnFailure = el.prop('checked'));
    bindControl('historyDepth', el => el.val(settings.historyDepth), el => settings.historyDepth = Math.min(40, Math.max(1, Number(el.val()) || 16)));
    bindControl('messageCharLimit', el => el.val(settings.messageCharLimit), el => settings.messageCharLimit = Math.max(50, Number(el.val()) || 400));
    bindControl('memberInfo', el => el.val(settings.memberInfo), el => settings.memberInfo = bindSelectValue('memberInfo'));
    bindControl('memberInfoCharLimit', el => el.val(settings.memberInfoCharLimit), el => settings.memberInfoCharLimit = Math.max(0, Number(el.val()) || 300));
    bindControl('stopBetweenMs', el => el.val(settings.stopBetweenMs), el => settings.stopBetweenMs = Math.max(0, Number(el.val()) || 0));
    bindControl('extraPrompt', el => el.val(settings.extraPrompt), el => settings.extraPrompt = String(el.val()));

    // nested strategies
    for (const key of ['natural', 'pool', 'list', 'manual']) {
        const id = `strategies_${key}`;
        bindControl(id, el => el.prop('checked', Boolean(settings.strategies[key])), el => settings.strategies[key] = el.prop('checked'));
    }
}

function addSettingsUI() {
    $('#extensions_settings').append(SETTINGS_HTML);
    if (!getElement('enabled').length) {
        console.error('[SceneDirector] settings panel did not attach to #extensions_settings');
        return;
    }
    bindElements();
    applyOutLabels();
    getElement('testTool').on('click', () => { void runToolingSelfTest(); });
    getElement('clearLog').on('click', () => {
        state.logLines = [];
        getElement('log').val('');
    });
}

// #endregion

// #region Slash command

function initSlashCommand() {
    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'director',
        helpString: 'Control the Group Scene Director. Usage: /director on|off|debug|mode chained|batch|status',
        unnamedArgumentList: [
            SlashCommandArgument.fromProps({
                description: 'Action',
                typeList: [ARGUMENT_TYPE.STRING],
                isRequired: true,
                defaultValue: 'status',
                enumList: ['on', 'off', 'debug', 'mode', 'status'],
            }),
        ],
        callback: async (_args, value) => {
            const action = String(value ?? 'status').trim().toLowerCase();
            switch (action) {
                case 'on':
                case 'off': {
                    settings.enabled = action === 'on';
                    saveSettings();
                    getElement('enabled').prop('checked', settings.enabled);
                    return `Scene Director ${action}`;
                }
                case 'debug':
                    console.log('[SceneDirector] payload', state.lastPayload);
                    console.log('[SceneDirector] decision', state.lastDecision);
                    console.log('[SceneDirector] error', state.lastError);
                    console.log('[SceneDirector] test', state.lastTest);
                    return `lastError: ${state.lastError ?? '(none)'} | lastTest: ${state.lastTest ? (state.lastTest.ok ? `PASSED (${state.lastTest.ms}ms)` : 'FAILED') : '(not run)'}`;
                case 'mode':
                    settings.mode = settings.mode === 'chained' ? 'batch' : 'chained';
                    saveSettings();
                    getElement('mode').val(settings.mode);
                    return `mode = ${settings.mode}`;
                default: {
                    const g = getGroup();
                    return `enabled=${settings.enabled}, mode=${settings.mode}, group=${g?.id ?? '(none)'}, busy=${isBusy()}`;
                }
            }
        },
    }));
}

// #endregion

// #region Lifecycle

export async function init() {
    addSettingsUI();
    initSlashCommand();
    injectGroupEditControls();

    eventSource.makeFirst(event_types.GENERATION_AFTER_COMMANDS, onGenerationAfterCommands);
    eventSource.on(event_types.GROUP_WRAPPER_STARTED, onGroupWrapperStarted);
    eventSource.on(event_types.GROUP_WRAPPER_FINISHED, onGroupWrapperFinished);
    eventSource.on(event_types.GROUP_UPDATED, syncGroupEditControls);
    eventSource.on(event_types.GENERATION_STOPPED, () => {
        const turn = state.turn;
        logDir(`Generation stopped (turn=${turn ? turn.mode + '/' + turn.phase : 'none'})`, { level: 'warn' });
        restoreOnAbandon();
    });

    logDir('Scene Director initialized. Use the Test & Debug section to verify tool calling support.');
    console.log('[SceneDirector] initialized');
    return true;
}

// Extensions are loaded as plain type=module <script> tags and are never
// invoked manually - the module must bootstrap itself (jQuery waits for DOM).
jQuery(async () => {
    try {
        await init();
    } catch (error) {
        console.error('[SceneDirector] init failed', error);
        toastr.error(`Scene Director failed to start: ${error?.message ?? error}`, 'Scene Director');
    }
});

// #endregion
