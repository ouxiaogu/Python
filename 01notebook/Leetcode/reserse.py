def reverse(x):
    """
    :type x: int
    :rtype: int
    """
    absx = abs(x)
    if absx > (1<<31)-1:
        return 0
    sign = -1
    if x>=0:
        sign = 1
    res = list(str(absx))
    res = res[::-1]
    res = ''.join(res)
    if sign*int(res) > (1<<31)-1:
        return 0
    return sign*int(res)

print reverse(1534236469)
print reverse(10)


raise TypeError