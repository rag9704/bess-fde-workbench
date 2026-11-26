import time

from ommx.v1 import DecisionVariable, Instance, State
from ommx_highs_adapter import OMMXHighsAdapter


def build(n: int) -> Instance:
    xs = [DecisionVariable.continuous(i, lower=0, upper=1) for i in range(n)]
    return Instance.from_components(
        decision_variables=xs,
        objective=sum(x for x in xs),
        constraints=[],
        sense=Instance.MAXIMIZE,
    )


for n in (2_000, 4_000, 8_000, 16_000):
    adapter = OMMXHighsAdapter(build(n))
    model = adapter.solver_input
    model.setOptionValue("output_flag", False)
    model.run()

    t0 = time.perf_counter()
    state = adapter.decode_to_state(model)
    t_decode = time.perf_counter() - t0

    t0 = time.perf_counter()
    col_value = model.getSolution().col_value  # one copy
    fixed = State(entries={v.id: col_value[i] for i, v in enumerate(adapter.instance.used_decision_variables)})
    t_fixed = time.perf_counter() - t0

    assert fixed.entries == state.entries
    print(f"n={n:>6}  decode_to_state {t_decode:7.3f}s   single-copy {t_fixed:7.4f}s   ({t_decode / t_fixed:,.0f}x)")
