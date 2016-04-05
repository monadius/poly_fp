from poly_fp import *

# Create variables
x = mk_var_expr("x")
y = mk_var_expr("y")

# Create expressions
expr1 = '1.1' * x - y
expr2 = (x * y - '2.1') * ('0.1' + x)

# The following expression is wrong:
# expr3 = ('1.1' + '1') * x
# '1.1' + '1' yields '1.11' in Python
# Here is the correct version:
expr3 = ('1.1' + mk_const_expr('1')) * x

analyze_float(expr1)
analyze_fixed(expr1)

# Change names of printed constants 'eps' and 'delta'
set_eps_name('e')
set_delta_name('d')

# Change the printed form of the absolute value function
set_abs_template('abs({0})')

analyze_float(expr2)
analyze_fixed(expr2)

# Assume now that all variables are real numbers and must be rounded
set_real_vars_flag(True)

analyze_float(expr1)
