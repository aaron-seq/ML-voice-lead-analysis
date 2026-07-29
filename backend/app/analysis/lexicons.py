"""Sales-conversation lexicons.

This module is the engine's domain knowledge, kept as data so it can be
reviewed, tested and tuned by someone who knows sales but not Python. Each
phrase list is deliberately narrow: precision matters more than recall here,
because a false "budget confirmed" signal is far more damaging to a rep's
trust than a missed one.

Phrases are matched on a normalised, lower-cased token stream with word
boundaries, so "cost" does not fire inside "costume". Multi-word phrases are
matched as contiguous token sequences.
"""

from __future__ import annotations

from app.domain.models import ObjectionCategory, SignalType

# --------------------------------------------------------------------------
# Sentiment
# --------------------------------------------------------------------------
# Weights are on [-1, 1]. Tuned for business conversation rather than product
# reviews -- "interesting" is mildly positive on a sales call, and "fine" is
# closer to neutral-negative than a general-purpose lexicon would suggest.

POSITIVE_TERMS: dict[str, float] = {
    "amazing": 0.9, "excellent": 0.85, "fantastic": 0.85, "perfect": 0.85,
    "brilliant": 0.8, "outstanding": 0.8, "impressive": 0.75, "great": 0.7,
    "love": 0.75, "excited": 0.75, "exciting": 0.7, "wonderful": 0.75,
    "helpful": 0.6, "useful": 0.55, "valuable": 0.65, "powerful": 0.55,
    "good": 0.5, "nice": 0.45, "solid": 0.45, "promising": 0.55,
    "interested": 0.6, "interesting": 0.4, "compelling": 0.65,
    "exactly": 0.5, "absolutely": 0.55, "definitely": 0.5, "certainly": 0.45,
    "agree": 0.5, "agreed": 0.55, "makes sense": 0.5, "sounds good": 0.6,
    "looking forward": 0.65, "happy": 0.6, "glad": 0.5, "pleased": 0.55,
    "yes": 0.3, "sure": 0.25, "clear": 0.35, "straightforward": 0.4,
    "easy": 0.45, "smooth": 0.45, "seamless": 0.55, "efficient": 0.5,
    "win": 0.6, "improve": 0.45, "better": 0.4, "best": 0.55,
}

NEGATIVE_TERMS: dict[str, float] = {
    "terrible": -0.9, "awful": -0.85, "horrible": -0.85, "useless": -0.8,
    "disappointing": -0.75, "frustrating": -0.7, "frustrated": -0.7,
    "annoying": -0.6, "confusing": -0.6, "confused": -0.55, "unclear": -0.5,
    "difficult": -0.5, "hard": -0.4, "complicated": -0.55, "complex": -0.35,
    "expensive": -0.6, "costly": -0.6, "overpriced": -0.8, "pricey": -0.55,
    "problem": -0.5, "issue": -0.4, "concern": -0.45, "concerned": -0.5,
    "worried": -0.55, "risk": -0.4, "risky": -0.55, "doubt": -0.5,
    "no": -0.3, "not": -0.25, "never": -0.5, "nothing": -0.35,
    "disagree": -0.6, "wrong": -0.55, "bad": -0.6, "worse": -0.6,
    "slow": -0.45, "broken": -0.7, "fail": -0.65, "failed": -0.65,
    "waste": -0.7, "unfortunately": -0.5, "sorry": -0.3, "afraid": -0.4,
    "busy": -0.3, "swamped": -0.4, "stretched": -0.35,
}

INTENSIFIERS: dict[str, float] = {
    "very": 1.4, "really": 1.35, "extremely": 1.6, "incredibly": 1.6,
    "so": 1.25, "quite": 1.15, "particularly": 1.25, "especially": 1.25,
    "absolutely": 1.5, "totally": 1.4, "completely": 1.45, "hugely": 1.5,
    "massively": 1.5, "super": 1.35,
}

DOWNTONERS: dict[str, float] = {
    "slightly": 0.6, "somewhat": 0.65, "a bit": 0.6, "a little": 0.6,
    "kind of": 0.6, "sort of": 0.6, "fairly": 0.8, "relatively": 0.8,
    "maybe": 0.65, "perhaps": 0.7, "possibly": 0.7,
}

NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
    "cannot", "cant", "wont", "wouldnt", "shouldnt", "isnt", "arent",
    "wasnt", "werent", "dont", "doesnt", "didnt", "hardly", "barely",
    "scarcely", "without", "lack", "lacks", "lacking",
})

# How far a negator's influence reaches forward, in tokens.
NEGATION_WINDOW = 3

# --------------------------------------------------------------------------
# Buying signals
# --------------------------------------------------------------------------
# Each entry: (phrase, strength). Strength reflects how diagnostic the phrase
# is of genuine buying intent, not how enthusiastic it sounds. "what's the
# pricing" is a stronger buying signal than "this is great".

SIGNAL_PHRASES: dict[SignalType, list[tuple[str, float]]] = {
    SignalType.BUDGET: [
        ("we have budget", 0.95), ("budget is approved", 0.95),
        ("budget approved", 0.95), ("we've allocated", 0.9),
        ("we have allocated", 0.9), ("set aside budget", 0.9),
        ("budget for this", 0.75), ("within our budget", 0.8),
        ("what does it cost", 0.6), ("what's the cost", 0.6),
        ("how much does it cost", 0.6), ("how much is it", 0.6),
        ("pricing", 0.5), ("price point", 0.55), ("your rates", 0.55),
        ("quote", 0.6), ("proposal", 0.6), ("contract", 0.7),
        ("purchase order", 0.85), ("procurement", 0.7),
    ],
    SignalType.AUTHORITY: [
        ("i can approve", 0.95), ("i'm the decision maker", 0.95),
        ("i am the decision maker", 0.95), ("my decision", 0.85),
        ("i own this", 0.8), ("i sign off", 0.9), ("i'll sign", 0.85),
        ("my team and i decide", 0.85), ("i'll bring in", 0.6),
        ("loop in my", 0.6), ("introduce you to", 0.6),
        # Naming the approver is a positive qualification signal even though
        # it is not the speaker: it means the buying committee is now known.
        ("our cto", 0.5), ("our cfo", 0.5), ("our ceo", 0.5),
        ("our vp", 0.5), ("head of", 0.4),
        ("would need to approve", 0.6), ("needs to approve", 0.6),
        ("has to approve", 0.6), ("sign off", 0.55),
        ("architecture committee", 0.5), ("steering committee", 0.5),
    ],
    SignalType.NEED: [
        ("we're struggling with", 0.9), ("we are struggling with", 0.9),
        ("our biggest problem", 0.9), ("pain point", 0.85),
        ("we need", 0.7), ("we really need", 0.85), ("looking for a solution", 0.85),
        ("trying to solve", 0.8), ("this is a problem for us", 0.85),
        ("costing us", 0.85), ("wasting time", 0.75), ("manual process", 0.7),
        ("doesn't scale", 0.75), ("does not scale", 0.75),
        ("falls over", 0.7), ("current process", 0.5),
    ],
    SignalType.TIMELINE: [
        ("this quarter", 0.85), ("next quarter", 0.7), ("this month", 0.85),
        ("by end of", 0.75), ("deadline", 0.8), ("go live", 0.8),
        ("start next week", 0.9), ("as soon as possible", 0.85),
        ("timeline", 0.6), ("when can we start", 0.9),
        ("how quickly can", 0.8), ("how soon", 0.75),
        ("kick off", 0.8), ("onboarding", 0.7), ("implementation", 0.6),
        # Naming a concrete day is the strongest everyday commitment device
        # there is, and it was previously invisible to the engine.
        ("tomorrow", 0.75), ("next week", 0.7), ("this week", 0.75),
        ("monday", 0.65), ("tuesday", 0.65), ("wednesday", 0.65),
        ("thursday", 0.65), ("friday", 0.65),
        ("q1", 0.7), ("q2", 0.7), ("q3", 0.7), ("q4", 0.7),
    ],
    SignalType.URGENCY: [
        ("urgent", 0.9), ("urgently", 0.9), ("critical", 0.8),
        ("top priority", 0.9), ("high priority", 0.85),
        ("can't wait", 0.75), ("running out of time", 0.9),
        ("contract expires", 0.9), ("renewal is coming", 0.85),
        ("need this yesterday", 0.9),
    ],
    SignalType.COMPETITOR: [
        ("currently using", 0.75), ("we use", 0.5), ("we're using", 0.6),
        ("compared to", 0.6), ("versus", 0.55), ("alternative", 0.55),
        ("evaluating", 0.7), ("shortlist", 0.8), ("rfp", 0.8),
        ("other vendors", 0.75), ("another vendor", 0.75),
        ("switching from", 0.85), ("migrate from", 0.8),
    ],
    SignalType.ENTHUSIASM: [
        ("this is exactly", 0.9), ("exactly what we need", 0.95),
        ("that's impressive", 0.8), ("i love", 0.8), ("we love", 0.8),
        ("this is great", 0.7), ("really like", 0.7),
        ("game changer", 0.85), ("blown away", 0.85),
        ("looking forward", 0.65),
    ],
    SignalType.NEXT_STEP: [
        ("next steps", 0.9), ("send me", 0.75), ("send over", 0.75),
        ("follow up", 0.7), ("schedule a", 0.85), ("book a", 0.85),
        ("set up a call", 0.9), ("put time in", 0.85),
        ("calendar invite", 0.9), ("demo", 0.7), ("trial", 0.8),
        ("pilot", 0.85), ("proof of concept", 0.85), ("poc", 0.8),
        ("get started", 0.85), ("move forward", 0.85),
    ],
    SignalType.TECHNICAL_FIT: [
        ("integrate with", 0.75), ("integration", 0.6), ("api", 0.55),
        ("does it support", 0.7), ("can it handle", 0.7),
        ("our stack", 0.65), ("sso", 0.7), ("single sign on", 0.7),
        ("security review", 0.8), ("soc 2", 0.8), ("gdpr", 0.7),
        ("data residency", 0.75), ("on premise", 0.7),
    ],
    SignalType.SOCIAL_PROOF: [
        ("case study", 0.7), ("reference customer", 0.85),
        ("who else uses", 0.8), ("similar company", 0.7),
        ("in our industry", 0.65), ("testimonial", 0.6),
    ],
}

# --------------------------------------------------------------------------
# Objections
# --------------------------------------------------------------------------

OBJECTION_PHRASES: dict[ObjectionCategory, list[tuple[str, float]]] = {
    ObjectionCategory.PRICE: [
        ("too expensive", 0.95), ("out of our budget", 0.95),
        ("no budget", 0.95), ("can't afford", 0.9), ("cannot afford", 0.9),
        ("price is high", 0.85), ("bit pricey", 0.8), ("costs too much", 0.9),
        ("cheaper", 0.7), ("discount", 0.6), ("budget is tight", 0.85),
        ("budget constraints", 0.85), ("hard to justify", 0.8),
    ],
    ObjectionCategory.TIMING: [
        ("not right now", 0.9), ("not a good time", 0.9),
        ("maybe next year", 0.85), ("next fiscal", 0.8),
        ("circle back", 0.7), ("revisit in", 0.75),
        ("too busy", 0.7), ("bandwidth", 0.6), ("other priorities", 0.85),
        ("after the", 0.4), ("put a pin in", 0.8),
    ],
    ObjectionCategory.AUTHORITY: [
        ("need to check with", 0.9), ("talk to my boss", 0.9),
        ("run it by", 0.85), ("not my decision", 0.95),
        ("i don't decide", 0.9), ("committee", 0.75),
        ("board approval", 0.85), ("need sign off", 0.85),
        ("above my pay grade", 0.9),
    ],
    ObjectionCategory.COMPETITOR: [
        ("already have", 0.8), ("happy with our current", 0.9),
        ("locked into", 0.9), ("under contract", 0.85),
        ("just signed with", 0.95), ("your competitor", 0.8),
        ("cheaper elsewhere", 0.85),
    ],
    ObjectionCategory.NEED: [
        ("don't see the value", 0.9), ("do not see the value", 0.9),
        ("not sure we need", 0.9), ("works fine", 0.75),
        ("not a priority", 0.9), ("doesn't apply to us", 0.85),
        ("not relevant", 0.8), ("we're fine", 0.7),
    ],
    ObjectionCategory.TRUST: [
        ("never heard of", 0.8), ("how do i know", 0.7),
        ("any references", 0.6), ("what if you", 0.5),
        ("security concerns", 0.85), ("compliance", 0.6),
        ("data privacy", 0.7), ("startup risk", 0.8),
    ],
    ObjectionCategory.COMPLEXITY: [
        ("too complicated", 0.9), ("steep learning curve", 0.85),
        ("hard to implement", 0.85), ("migration would be", 0.75),
        ("training our team", 0.6), ("change management", 0.7),
        ("disruptive", 0.75),
    ],
}

# Playbook guidance surfaced with each objection. Static, reviewed copy --
# deliberately not model-generated, so a rep always sees advice a sales leader
# has signed off on.
OBJECTION_PLAYBOOK: dict[ObjectionCategory, str] = {
    ObjectionCategory.PRICE: (
        "Reframe from cost to cost-of-inaction. Quantify the pain they described "
        "earlier in the call, then anchor pricing against that number. Avoid "
        "discounting before the value case is agreed."
    ),
    ObjectionCategory.TIMING: (
        "Timing objections are usually priority objections. Ask what would have "
        "to be true for this to move up the list, and agree a specific date to "
        "revisit with a calendar invite before the call ends."
    ),
    ObjectionCategory.AUTHORITY: (
        "Do not treat this as a blocker -- treat it as a routing instruction. "
        "Ask who else is involved, what they care about, and offer to join the "
        "internal conversation with material tailored to them."
    ),
    ObjectionCategory.COMPETITOR: (
        "Establish what is and is not working with the incumbent before "
        "differentiating. Ask about the renewal date and the switching cost; "
        "an unhappy customer under contract is a dated opportunity, not a loss."
    ),
    ObjectionCategory.NEED: (
        "Value objections mean discovery was incomplete. Return to their "
        "current process and quantify what it costs them today. Do not present "
        "features until a problem is agreed."
    ),
    ObjectionCategory.TRUST: (
        "Offer proof rather than assurance: reference customers of similar size, "
        "security documentation, or a time-boxed paid pilot that caps their risk."
    ),
    ObjectionCategory.COMPLEXITY: (
        "Shrink the perceived first step. Propose a narrow pilot with a named "
        "success metric and share an implementation timeline from a comparable "
        "customer."
    ),
}

# --------------------------------------------------------------------------
# Outcome detection
# --------------------------------------------------------------------------

OUTCOME_PHRASES: dict[str, list[tuple[str, float]]] = {
    "meeting_booked": [
        ("i'll send an invite", 0.95), ("send you an invite", 0.95),
        ("calendar invite", 0.9), ("book it in", 0.9),
        ("see you on", 0.85), ("talk to you on", 0.8),
        ("schedule a demo", 0.9), ("put it in the calendar", 0.95),
        ("does tuesday work", 0.85), ("does thursday work", 0.85),
        ("next week works", 0.85), ("meeting next", 0.8),
        # Agreeing to a proposed slot is how meetings actually get booked in
        # conversation; the phrasing above only covered the rep's side.
        ("monday works", 0.9), ("tuesday works", 0.9),
        ("wednesday works", 0.9), ("thursday works", 0.9),
        ("friday works", 0.9), ("that works", 0.7),
        ("works for me", 0.8), ("let's do that", 0.7),
        ("lets do that", 0.7), ("put fifteen minutes in", 0.85),
        ("put some time in", 0.85), ("get something in the diary", 0.9),
    ],
    "follow_up_agreed": [
        ("i'll follow up", 0.85), ("follow up next", 0.85),
        ("check back in", 0.8), ("touch base", 0.8),
        ("reach out", 0.6), ("circle back", 0.75), ("get back to you", 0.75),
    ],
    "information_requested": [
        ("send me the", 0.85), ("send over the", 0.85),
        ("email me", 0.8), ("share the deck", 0.85),
        ("send the pricing", 0.9), ("send a proposal", 0.9),
        ("documentation", 0.6), ("case study", 0.6),
    ],
    "not_interested": [
        ("not interested", 0.95), ("no thank you", 0.85),
        ("please remove me", 0.95), ("stop calling", 0.95),
        ("take me off", 0.9), ("don't call", 0.9),
    ],
    "not_a_fit": [
        ("not a fit", 0.9), ("doesn't fit", 0.85),
        ("not what we're looking for", 0.9), ("too small for us", 0.85),
        ("too big for us", 0.85), ("wrong department", 0.8),
    ],
    "gatekeeper_only": [
        ("i'm not the right person", 0.9), ("i am not the right person", 0.9),
        ("you want to speak to", 0.85), ("transfer you", 0.8),
        ("i just answer the phone", 0.9), ("i'll pass on the message", 0.85),
    ],
}

# --------------------------------------------------------------------------
# Discourse helpers
# --------------------------------------------------------------------------

# Phrases typical of the seller's side. Used to infer speaker roles when the
# transcript does not label them (see analysis.diarization).
REP_MARKERS: frozenset[str] = frozenset({
    "reaching out", "calling from", "i'm calling", "quick call",
    "our platform", "our product", "our solution", "we help companies",
    "what we do", "let me show you", "let me walk you through",
    "does that make sense", "happy to send", "i'll send you",
    "book some time", "great question", "absolutely", "just to confirm",
})

PROSPECT_MARKERS: frozenset[str] = frozenset({
    "we currently", "our team", "we're using", "we use",
    "how much", "what's the price", "send me", "i'll need to",
    "we already have", "not interested", "our budget",
    "who else uses", "can it", "does it",
})

# Words with no analytical value, removed before topic extraction.
#
# Note the collapsed contraction forms (thats, ill, weve, dont ...). Tokenising
# strips apostrophes so "that's" arrives as "thats"; without these entries the
# most common words in any transcript surface as its top "topics".
STOPWORDS: frozenset[str] = frozenset(
    """
a about above after again against all am an and any are as at be because
been before being below between both but by can cannot could did do does
doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours
ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves yeah yep ok okay right like know
think going get got really actually basically kind sort thing things stuff
lot bit sure mean say said talk see look well one two make made take give
thats im ive ill id youre youve youll youd hes shes theyre theyve theyll
weve wed dont doesnt didnt cant couldnt wont wouldnt shouldnt isnt arent
wasnt werent hasnt havent hadnt lets thatll theres heres gonna wanna gotta
yes nope maybe also much many every
""".split()
)

FILLER_TOKENS: frozenset[str] = frozenset({
    "um", "uh", "erm", "ah", "eh", "hmm", "mm", "mhm", "uh-huh", "you know",
    "i mean", "like", "sort of", "kind of",
})
