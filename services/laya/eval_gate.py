import statistics
import time

import laya_mlx as laya

MATCHING = [
    ("Book a meeting with Sara tomorrow at 3pm.", "book_meeting(person='Sara', date='tomorrow', time='15:00')"),
    ("Can you remind me to call my mom at 6 tonight?", "set_reminder(text='call mom', time='18:00 today')"),
    ("Send John an email saying I'll be late.", "send_email(to='John', body='I will be late')"),
    ("Transfer 50 euros to Marco.", "transfer_money(to='Marco', amount=50, currency='EUR')"),
    ("Cancel my 10am meeting.", "cancel_meeting(time='10:00')"),
    ("Please move the dentist appointment to Friday.", "reschedule(event='dentist appointment', date='Friday')"),
    ("Order a large pepperoni pizza.", "order_food(item='pepperoni pizza', size='large')"),
    ("Turn off the living room lights.", "set_lights(room='living room', on=False)"),
    ("Add milk and eggs to my shopping list.", "add_to_list(list='shopping', items=['milk', 'eggs'])"),
    ("Text Anna that dinner is at eight.", "send_message(to='Anna', text='dinner is at eight')"),
    ("Pay the electricity bill of 120 dollars.", "pay_bill(payee='electricity', amount=120, currency='USD')"),
    ("Schedule a call with the design team next Monday morning.", "book_meeting(person='design team', date='next Monday', time='09:00')"),
]

WRONG_ARGUMENTS = [
    ("Book a meeting with Sara tomorrow at 3pm.", "book_meeting(person='Sara', date='tomorrow', time='13:00')"),
    ("Transfer 50 euros to Marco.", "transfer_money(to='Marco', amount=500, currency='EUR')"),
    ("Send John an email saying I'll be late.", "send_email(to='Joan', body='I will be late')"),
    ("Remind me to call my mom at 6 tonight.", "set_reminder(text='call mom', time='06:00 tomorrow')"),
    ("Turn off the living room lights.", "set_lights(room='bedroom', on=False)"),
    ("Pay the electricity bill of 120 dollars.", "pay_bill(payee='electricity', amount=120, currency='EUR')"),
]

OPPOSITE_INTENT = [
    ("Don't cancel my 10am meeting, just move it to 11.", "cancel_meeting(time='10:00')"),
    ("Turn on the living room lights.", "set_lights(room='living room', on=False)"),
    ("Don't send that email yet.", "send_email(to='John', body='I will be late')"),
]

UNRELATED = [
    ("What's the weather like tomorrow?", "book_meeting(person='Sara', date='tomorrow', time='15:00')"),
    ("Read me my last message from Anna.", "send_message(to='Anna', text='dinner is at eight')"),
    ("I was just wondering how much I spent this month.", "transfer_money(to='Marco', amount=50, currency='EUR')"),
]

CASES = [(said, action, True) for said, action in MATCHING] + [
    (said, action, False) for said, action in WRONG_ARGUMENTS + OPPOSITE_INTENT + UNRELATED
]

INSTRUCTIONS = "Does the proposed action do exactly what the user asked, with the same details?"


def state(said: str, action: str) -> str:
    return f"User request: {said}\nProposed action: {action}"


QUESTIONS = {
    "noul": {"type": "noul", "instructions": INSTRUCTIONS},
    "choice": {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {
            "A": "Yes: the action matches the request, including every detail.",
            "B": "No: the action differs from the request (wrong details, opposite intent, or not requested).",
        },
    },
}


def p_yes(answer: dict, kind: str) -> float:
    if kind == "noul":
        return float(answer["noul"])
    probs = answer.get("probabilities") or answer.get("distribution") or {}
    return float(probs.get("A", 1.0 if answer.get("choice") == "A" else 0.0))


def evaluate(repo: str) -> None:
    agent = laya.load(repo)
    agent.predict(state(*CASES[0][:2]), QUESTIONS)
    print(f"\n=== {repo}")
    first = True
    for kind in QUESTIONS:
        correct, latencies, rows = 0, [], []
        for said, action, label in CASES:
            t = time.perf_counter()
            result = agent.predict(state(said, action), {kind: QUESTIONS[kind]})
            latencies.append((time.perf_counter() - t) * 1000)
            answer = result["answers"][kind]
            if first:
                print("  answer shape:", answer)
                first = False
            p = p_yes(answer, kind)
            ok = (p >= 0.5) == label
            correct += ok
            rows.append((ok, p, label, said, action))
        print(f"  [{kind}] accuracy {correct}/{len(CASES)}   p50 {statistics.median(latencies):.1f}ms")
        for ok, p, label, said, action in rows:
            if not ok:
                print(f"     ✗ p(yes)={p:.2f} expected {'yes' if label else 'no'}: {said} → {action}")


for repo in ("aac6fef/laya-mlx", "aac6fef/laya-multilingual-mlx"):
    evaluate(repo)
