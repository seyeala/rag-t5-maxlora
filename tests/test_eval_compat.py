import eval.eval_instruction as compat_instruction
import eval.eval_sst2 as compat_sst2
import rag_t5.eval.instruction as canonical_instruction
import rag_t5.eval.sst2 as canonical_sst2


def test_eval_compatibility_wrappers():
    assert compat_instruction.evaluate is canonical_instruction.evaluate
    assert compat_sst2.evaluate is canonical_sst2.evaluate
