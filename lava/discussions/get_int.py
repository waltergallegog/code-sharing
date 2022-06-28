from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var

import numpy as np
from lava.magma.core.model.py.model                import PyLoihiProcessModel
from lava.magma.core.decorator                     import implements, requires, tag
from lava.magma.core.resources                     import CPU
from lava.magma.core.model.py.type                 import LavaPyType
from lava.magma.core.model.py.ports                import PyInPort, PyOutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

# Minimal process with an OutPort
class P1(AbstractProcess):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.out = OutPort(shape=(1,))


# Minimal process with an InPort
class P2(AbstractProcess):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.inp = InPort(shape=(1,))
		self.in_data = Var(shape=(1,), init=0)
		
		
# A minimal PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModelA(PyLoihiProcessModel):
	out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

	def run_spk(self):
		data = np.array([5])
		# print("Sent output data of P1: {}".format(data))
		self.out.send(data)
		

# A minimal PyProcModel implementing P2
@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyProcModelB(PyLoihiProcessModel):
	inp    : PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
	in_data: int      = LavaPyType(int, int)

	def run_spk(self):
		self.in_data = self.inp.recv()
		dummy = self.in_data[0]
		print(f"Received input data in P2: {dummy}")
		print(f"Received input type in P2: {type(dummy)}")
		

def run_and_stop():
	sender = P1()
	recv   = P2()
	
	# Connecting output port to an input port
	sender.out.connect(recv.inp)
	
	sender.run(RunSteps(num_steps=1), Loihi1SimCfg())
	recv_data = recv.in_data.get()[0]
	print(f"Received input data in main: {recv_data}")
	print(f"Received input type in main: {type(recv_data)}")
	
	sender.stop()
	recv  .stop()
		
		
if __name__ == '__main__':
	run_and_stop()