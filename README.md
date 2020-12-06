# poly_fp
Symbolic round-off errors (floating-point and fixed-point) for general polynomial expressions.

See [examples.py](examples.py) for usage examples. 
See also the [source code](poly_fp.py) for additional comments.

## Output format

Main functions are `analyze_float(expr)` and `analyze_fixed(expr)`. These functions print round-off error bounds for given expressions. 

The output format of `analyze_float(expr)` is the following.
First, the real-valued part of the expression is printed (it is called `v0`). Then the round-off error is divided into two parts: `err_rel` and `err_abs`. For each part, two error bounds are printed: the first bound is more precise (and complex) and the second bound is less precise but simpler (in general). The `err_rel` part is further divied into two parts: `err_rel1` (linear part) and `err_rel2` (higher order part). One error bound is printed for `err_rel1` and two error bounds (more accurate and less accurate) are printed for `err_rel2`. In general, the sum of bounds of `err_rel1` and `err_rel2` is more precise than direct bounds of `err_rel`.

Output example:

    float(((11/10) * (x)) - (y)) = v0 + error
    v0 = 11/10*x + -1*y

    error = err_rel + err_abs

    |err_rel| <= |11/10*x| * ((1 + eps)^3 - 1) + |-1*y| * ((1 + eps)^1 - 1)
    |err_rel| <= (|33/10*x| + |-1*y|) * eps / (1 - 3*eps)

    |err_abs| <= |1| * delta^1 * (1 + eps)^1
    |err_abs| <= (|1|) * (1 + eps)^1 * delta

    err_rel = err_rel1 + err_rel2

    |err_rel1| <= (|11/10*x| + |11/10*x| + |11/10*x + -1*y|) * eps
    |err_rel2| <= |11/10*x| * ((1 + eps)^3 - 1 - 3*eps)
    |err_rel2| <= (|99/10*x|) * eps^2 / (1 - 3*eps)

The constants `eps` and `delta` represent maximum relative and absolute errors of the standard floating-point rounding model: `rnd(t) = t * (1 + e) + d`.

Values of these constants can be defined as follows.

- single precision (nearest): `eps = 2^(-24)`, `delta = 2^(-150)`
- double precision (nearest): `eps = 2^(-53)`, `delta = 2^(-1075)`
- single precision (directed): `eps = 2^(-23)`, `delta = 2^(-149)`
- double precision (directed): `eps = 2^(-52)`, `delta = 2^(-1074)`

The output format of the fixed-precision analysis is similar but it does not include relative errors. The rounding model is defined by `rnd(t) = t + d` where the value of `d` depends on a chosen 
fixed-precision format. 
(For instance, if the format has k fractional bits, then d = 2^(-k) for simple rounding.)

## Limitations

It is assumed that all operations are performed with the same floating-point (or fixed-point) precision without overflows.

Constants which can be represented exactly by floating-point (fixed-point) numbers introduce round-off errors. This may be improved in future. Also, all constants are rounded as `rnd(c) = c * (1 + e)`. That is, we ignore subnormal constants (which are very rare).


## Main idea

The main idea of the program can be explained with a simple example. Suppose we have an expression

    p(x, y) = x + 0.1 * y

and we want to estimate the round-off error of its floating-point implementation (for example, using single precision arithmetic). We can write
    
    float(p(x, y)) = float(x + 0.1 * y) = rnd(x + rnd(rnd(0.1) * y)).

Here we assume that x and y are floating-point variables and we do not round their values. We apply the standard rounding model (with different error terms for different rounding operators) and get

    rnd(x + rnd(rnd(0.1) * y)) = (x + [0.1 * (1 + e1) + d1] * y * (1 + e2) + d2) * (1 + e3) + d3.

We can assume that `d1 = 0` (because `0.1` is a constant which can be rounded to a normal floating-point number; in fact, in our implementation we assume that all constants are rounded to normal floating-point numbers). We can also say that `d3 = 0` because addition and subtraction operations are always exact in the subnormal range. Finally, we get

    float(p(x, y)) = x * (1 + e3) + 0.1 * y * (1 + e1) * (1 + e2) * (1 + e3) + d2 * (1 + e3).

We can write

    float(p(x, y)) = x + 0.1 * y + err_rel + err_abs

where `err_abs` contains terms with variables `d` and `err_rel` contains all remaining terms. There several ways to estimate the computed round-off errors. A simple estimate is the following:

    |(1 + e1)(1 + e2)(1 + e3) - 1| <= (1 + eps)^3 - 1 <= 3*eps / (1 - 3*eps).

Our program yields the following output for this example:

    float((x) + ((1/10) * (y))) = v0 + error
    v0 = 1*x + 1/10*y

    error = err_rel + err_abs

    |err_rel| <= |1*x| * ((1 + eps)^1 - 1) + |1/10*y| * ((1 + eps)^3 - 1)
    |err_rel| <= (|1*x| + |3/10*y|) * eps / (1 - 3*eps)

    |err_abs| <= |1| * delta^1 * (1 + eps)^1
    |err_abs| <= (|1|) * (1 + eps)^1 * delta

    err_rel = err_rel1 + err_rel2

    |err_rel1| <= (|1/10*y| + |1/10*y| + |1*x + 1/10*y|) * eps
    |err_rel2| <= |1/10*y| * ((1 + eps)^3 - 1 - 3*eps)
    |err_rel2| <= (|9/10*y|) * eps^2 / (1 - 3*eps)

The bound of `err_rel1` is obtained in the following way:

    float(p(x, y)) = x + 0.1 * y + (0.1 * y * e1 + 0.1 * y * e2 + (x + 0.1 * y) * e3) + ...
    |0.1 * y * e1 + 0.1 * y * e2 + (x + 0.1 * y) * e3| <= (|0.1 * y| + |0.1 * y| + |x + 0.1 * y|) * eps