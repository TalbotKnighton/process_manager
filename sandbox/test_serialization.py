from process_manager import data_handlers as dh

rvhash = dh.RandomVariableHash()
a = dh.UniformDistribution(name='a', low=0, high=1).register_to_hash(rvhash, size=1)
print(a)
print(dh.UniformDistribution(name='a', low=0, high=1).sample(squeeze=True))
