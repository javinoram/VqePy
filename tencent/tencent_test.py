from tencirchem import UCCSD
from tencirchem.molecule import h2

uccsd = UCCSD(h2)
uccsd.kernel()
uccsd.print_summary(include_circuit=True)