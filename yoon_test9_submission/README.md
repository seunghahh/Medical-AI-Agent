# yoon test9 submission

N.O.V.A. preliminary-round agent based on the `hsh-test` branch runtime.

This version improves adaptive information gathering with bounded follow-up questions when answers are partial, and patient communication by addressing stated concerns, delivering concise evidence-grounded explanations, and responding to questions during the closing dialogue. It preserves evidence-linked SOAP notes and checks generated content against observed facts.

The package is standard-library-only. The runtime reads mounted guidelines from `NOVA_GUIDELINES_DIR` when available and submits through the provided NOVA client.

No score claim is made. The linked submission detail page was not directly accessible from the browsing tool; the public evaluation, rules, and developer-guide pages informed this build.
