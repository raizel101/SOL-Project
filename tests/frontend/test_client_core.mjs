// Pure browser logic, exercised under Node without a model, browser or DOM.
import assert from 'node:assert/strict';
import {
  countCharacters, validateMessage, validateHistory, historyAfterAccepted,
  classifyChatResponse, safeSourceUrl,
} from '../../frontend/src/client-core.js';

const accepted = {
  ok: true, error: null, mode: 'local_rag', answer: 'A bounded fixture reply. [S1]',
  llm_called: true, citations: [], source_passages: [], warnings: [],
};
const tests = [
  ['unicode_character_count_matches_python', () => {
    assert.equal(countCharacters('🙂'.repeat(1200)), 1200);
    assert.equal(validateMessage('🙂'.repeat(1200)).valid, true);
    assert.equal(validateMessage('🙂'.repeat(1201)).valid, false);
    assert.equal(countCharacters('हिन्दी'), [...'हिन्दी'].length);
  }],
  ['messages_are_nonempty_bounded_and_valid_unicode', () => {
    for (const text of [null, 1, '', '  \n', '\ud800', 'bad\u0000input']) {
      assert.equal(validateMessage(text).valid, false);
    }
    assert.equal(validateMessage('A real question\nwith a new line').valid, true);
  }],
  ['history_requires_complete_exact_role_content_pairs', () => {
    assert.equal(validateHistory([]).valid, true);
    const valid = [{ role: 'user', content: 'Question' }, { role: 'assistant', content: 'Reply' }];
    assert.equal(validateHistory(valid).valid, true);
    for (const history of [null, [valid[0]], [valid[1], valid[0]],
      [{ role: 'user', content: 'Q', tool: 'ignored' }, valid[1]],
      [{ role: 'user', content: '\ud800' }, valid[1]], Array(10).fill(valid[0])]) {
      assert.equal(validateHistory(history).valid, false);
    }
  }],
  ['history_caps_are_checked_without_trimming', () => {
    const each = [{ role: 'user', content: '🙂'.repeat(1000) }, { role: 'assistant', content: 'x'.repeat(1000) }];
    assert.equal(validateHistory(each).valid, true);
    assert.equal(validateHistory(each).total, 2000);
    assert.equal(validateHistory([...each, { role: 'user', content: 'x' }, { role: 'assistant', content: 'x' }]).valid, false);
    assert.equal(validateHistory([{ role: 'user', content: 'x' }, { role: 'assistant', content: 'x'.repeat(1001) }]).valid, false);
  }],
  ['completed_turn_keeps_all_text_and_does_not_mutate_prior_context', () => {
    const prior = [{ role: 'user', content: 'Old question' }, { role: 'assistant', content: 'Old reply' }];
    const snapshot = JSON.stringify(prior);
    const long = 'Full response'.repeat(100);
    const next = historyAfterAccepted(prior, 'New question', long);
    assert.equal(JSON.stringify(prior), snapshot);
    assert.equal(next.history.length, 4);
    assert.equal(next.history[3].content, long);
    assert.equal(next.validation.valid, false);
  }],
  ['only_recognized_valid_success_is_accepted', () => {
    assert.equal(classifyChatResponse(200, accepted).kind, 'accepted');
    for (const body of [null, [], {}, { ...accepted, error: {} }, { ...accepted, ok: false },
      { ...accepted, mode: 'unknown' }, { ...accepted, answer: '  ' }, { ...accepted, error: undefined }]) {
      assert.equal(classifyChatResponse(200, body).kind, 'error');
    }
    assert.equal(classifyChatResponse(503, accepted).kind, 'error');
  }],
  ['known_503_withholding_keeps_fixed_fallback_but_is_not_accepted', () => {
    for (const mode of ['source_claim_withheld', 'support_response_withheld', 'citation_check_failed', 'incomplete_response']) {
      const body = { ...accepted, ok: false, mode, answer: 'Fixed fallback, not rejected model text.',
        error: { code: mode, message: 'A fixture answer was withheld.', retryable: false } };
      const result = classifyChatResponse(503, body);
      assert.equal(result.kind, 'withheld');
      assert.equal(result.answer, body.answer);
      assert.equal(result.retryable, false);
    }
  }],
  ['unrecognized_errors_never_display_unaccepted_answer_text', () => {
    const result = classifyChatResponse(503, { ...accepted, ok: false, mode: 'error', answer: 'UNVERIFIED_FIXTURE_TEXT',
      error: { code: 'busy', message: 'Try again later.', retryable: true } });
    assert.equal(result.kind, 'error');
    assert.equal(result.answer, '');
    assert.equal(result.retryable, true);
  }],
  ['source_links_are_validated_instead_of_injected', () => {
    assert.equal(safeSourceUrl('https://incarnateword.in/cwm/12/education'), 'https://incarnateword.in/cwm/12/education');
    for (const value of ['javascript:alert(1)', 'data:text/html,hello', 'file:///C:/private.txt', '//host/path',
      'https://user:password@example.com', 'https://example.com\n', 'https://example.com/a b', null]) {
      assert.equal(safeSourceUrl(value), null);
    }
  }],
];
const failures = [];
for (const [name, run] of tests) {
  try { run(); } catch (error) { failures.push({ name, error: error.message }); }
}
process.stdout.write(JSON.stringify({ status: failures.length ? 'failed' : 'passed', tests_run: tests.length,
  passed: tests.length - failures.length, failures, model_calls: 0, browser_required: false }) + '\n');
process.exitCode = failures.length ? 1 : 0;
