# Nico
# 时间：2021/9/8 20:55

def func(*args, **kwargs):
    print(args)
    print(kwargs)

func(1, 2, 3, 4, x=3, y=5)