from poly_fp import *

# Create variables
x = mk_var_expr("x")
y = mk_var_expr("y")

# Create expressions
expr1 = -x
expr2 = '-1.1' + x
expr3 = -x * (-y)
expr4 = -(x + y)

print("\n*** expr1 ***")
analyze_float(expr1)
print("---------------")
analyze_fixed(expr1)

print("\n*** expr2 ***")
analyze_float(expr2)
print("---------------")
analyze_fixed(expr2)

print("\n*** expr3 ***")
analyze_float(expr3)
print("---------------")
analyze_fixed(expr3)

print("\n*** expr4 ***")
analyze_float(expr4)
print("---------------")
analyze_fixed(expr4)
