# agent-edge-layer

طبقة قرارات حتمية لحواف الرسم البياني للوكلاء، مبنية على مذكرة *Jev Engineering for Agent Graphs* (نوفمبر 2026).
بدل أن يعيد نموذج التخطيط قراءة الحالة كاملة عند كل حافة ويجيب بنص حر، تطرح كل حافة أسئلة محددة النوع في طلب واحد:

| المفتاح | النوع | السؤال |
|---|---|---|
| `agent` | Choice | أي وكيل يتولى الخطوة التالية؟ (`research`, `code`, `write`, `ask`) |
| `effort` | Score 1–5 | ما مقدار الجهد؟ ويحدد النموذج العامل وجهده |
| `safe.<risk>` | Noul | هل الفعل غير القابل للتراجع آمن؟ واحد لكل خطر: `delete`, `payment`, `outbound_email` |
| `pass` | Noul | هل مخرج العقدة يحقق شرط قبولها؟ |

## كيف يُتخذ القرار

1. تُرسل الأسئلة كلها إلى Jev في طلب واحد.
2. أي سؤال لم تبلغ ثقته عتبته يُرسل **وحده** إلى الطبقة الأعلى (Claude Opus 5.5).
3. إن بقي غير مؤكد في آخر طبقة تُطبَّق قيمة آمنة: التوجيه إلى `ask`، الجهد يُقرَّب للأعلى، الفعل الخطر يُعلَّق، والمخرج يُعاد تشغيله.

الأولوية في الخطة: رفض المخرج ← توجيه إلى `ask` ← تعليق فعل خطر ← تنفيذ.
نفس الاحتمالات تعطي دائماً نفس الخطة.

## الاستخدام

```python
import anthropic
from jev_edges import (DEFAULT_THRESHOLDS, ClaudeBackend, DecisionLog, EdgeDecisionLayer,
                       HopInput, JevBackend, NodeResult, TieredDecider, ToolCall)

decider = TieredDecider(
    [JevBackend(typesafe_client), ClaudeBackend(anthropic.Anthropic())],
    thresholds=DEFAULT_THRESHOLDS,
)
layer = EdgeDecisionLayer(decider)
log = DecisionLog("logs/edges.jsonl")

plan = layer.decide(HopInput(
    graph=graph_summary,
    finished=NodeResult("draft", output, "tests pass", attempt=1),
    pending_tool=ToolCall("send_email", {"to": "..."}),
))
log.record(hop_id, plan)
# plan.action: proceed | await_approval | rerun | halt_ask
# plan.next_agent, plan.worker.model, plan.worker.effort, plan.blocked_risks, plan.reasons
```

`JevBackend` يستدعي `client.decide(model=, state=, questions=)` كما في المذكرة. المذكرة تصف الكود بأنه مسودة،
لذلك مرّر `to_sdk` و`read_probs` لربطه بحقول TypeSafe الفعلية.

## الأمان

- قواعد الأفعال الخطرة وعتباتها في `jev_edges/safety.py` وحده، ومحمية بـ `CODEOWNERS`.
- الطبقة ترفض أن تُنشأ إذا كانت أي عتبة أمان أقل من `SAFE_YES_FLOOR` (0.90)، بما في ذلك نسيان ضبطها.
- `tests/test_safety.py` يفشل في CI إذا أُضعفت السياسة.
- سجل القرارات يحفظ الاحتمالات والقرارات فقط، ولا يحفظ حالة الرسم أو مدخلات الأدوات أو المخرجات.

## المعايرة

بعد تشغيل حقيقي، سجّل ما تبيّن أنه صحيح لكل قرار:

```python
log.record_outcome(hop_id, "agent", "research")   # الوكيل الذي كان يجب أن يعمل
log.record_outcome(hop_id, "safe.delete", "no")   # هل كان الفعل آمناً فعلاً
```

ثم:

```bash
python -m jev_edges.calibrate logs/edges.jsonl --tier jev --min-samples 30
```

تقترح الأداة لكل سؤال أقل عتبة تحقق الدقة المطلوبة (0.99 للأمان، 0.90 لغيره) ونسبة التصعيد الناتجة.
الاقتراحات لا تُطبَّق تلقائياً، وعتبة الأمان لا تنزل أبداً عن الحد الأدنى.

## التطوير

```bash
pip install -e ".[dev,claude]"
pytest -q
python example.py   # أربع قفزات محاكاة دون شبكة
```
