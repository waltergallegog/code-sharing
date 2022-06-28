import psutil

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

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
		shape = kwargs.get('shape', (1,))
		self.out = OutPort(shape=shape)


# Minimal process with an InPort
class P2(AbstractProcess):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		shape = kwargs.get('shape', (1,))
		self.inp = InPort(shape=shape)
		
		
# A minimal PyProcModel implementing P1
@implements(proc=P1, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModelA(PyLoihiProcessModel):
	out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

	def run_spk(self):
		data = np.array([1])
		# print("Sent output data of P1: {}".format(data))
		self.out.send(data)
		p = psutil.Process()
		print(f"PID P1: {p.pid} {p.open_files()}")


# A minimal PyProcModel implementing P2
@implements(proc=P2, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyProcModelB(PyLoihiProcessModel):
	inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)

	def run_spk(self):
		in_data = self.inp.recv()
		p = psutil.Process()
		print(f"PID P2: {p.pid} {p.open_files()}")
		
		# print("Received input data for P2: {}".format(in_data))
		

def run_and_stop():
	sender = P1()
	recv   = P2()
	
	# Connecting output port to an input port
	sender.out.connect(recv.inp)
	
	sender.run(RunSteps(num_steps=1), Loihi1SimCfg())
	sender.stop()
	recv  .stop()

def run_loop():
	p = psutil.Process()
	for _r in range(147):
		run_and_stop()
		print(f"PID MN: {p.pid} {p.open_files()}\n")
		
		
if __name__ == '__main__':
	run_loop()