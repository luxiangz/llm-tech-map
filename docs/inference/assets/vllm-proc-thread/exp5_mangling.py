"""实验 5：类体里的双下划线名字会被 name mangling 改写。

用来核对 `ShowHandsDSALayer._init_weights` 里那个 `def __load_weights(...)`。
关键点：它虽然是"局部函数"，但**文本上处在 class 体内**，所以照样被改写。
"""


class ShowHandsDSALayer:
    def _init_weights(self, model_path):
        def __load_weights(device_id, model_path):
            return f"loaded device {device_id} from {model_path}"

        def _runner(dev_id):
            return __load_weights(dev_id, model_path)

        return _runner(0)

    def __private_method(self):
        return "私有方法"


obj = ShowHandsDSALayer()
print("1) 调用结果:", obj._init_weights("/weights"))

code = ShowHandsDSALayer._init_weights.__code__
print("2) _init_weights 的局部变量表 co_varnames :", code.co_varnames)
print("   _init_weights 的闭包变量表 co_cellvars :", code.co_cellvars)

inner_names = {c.co_name: c for c in code.co_consts if hasattr(c, "co_name")}
runner = inner_names["_runner"]
print("3) _runner 的 co_freevars（它从外层捕获的名字）:", runner.co_freevars)
print("   嵌套函数的 co_name 仍是原名（co_name 不受改写）:",
      [n for n in inner_names if "load_weights" in n])

print("4) 类属性视角：", [n for n in vars(ShowHandsDSALayer) if "private" in n])
try:
    obj.__private_method()
except AttributeError as e:
    print("   obj.__private_method() -> AttributeError:", e)
print("   obj._ShowHandsDSALayer__private_method() ->",
      obj._ShowHandsDSALayer__private_method())
