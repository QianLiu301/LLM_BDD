from z3 import Solver, sat, unsat, Bool

s = Solver()
a, b = Bool('a'), Bool('b')
s.add(a, b)
print(s.check())