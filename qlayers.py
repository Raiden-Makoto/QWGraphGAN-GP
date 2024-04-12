import torch
import torch.nn as nn
import pennylane as qml

def QuantumDenseLayer(wires, backend="default.qubit.torch"):
    dev = qml.device(backend, wires)
    weight_shapes = {"weights": (1, wires, 3)}
    
    def PQC(inputs, weights):
        qml.templates.AmplitudeEmbedding(inputs, wires=range(wires), normalize=True)
        qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(wires)]

    QDL = qml.QNode(PQC, dev, interface="torch")
    return qml.qnn.TorchLayer(QDL, weight_shapes=weight_shapes)
    
class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, n_qlayers=1, backend="default.qubit.torch"):
        super(QLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_inputs = [f"wire_inputs_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_inputs = qml.device(self.backend, wires=self.wires_inputs)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        def _circuit_inputs(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_inputs)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_inputs, rotation=qml.RY)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_inputs]
        self.qlayer_inputs = qml.QNode(_circuit_inputs, self.dev_inputs, interface="torch")

        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]
        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        weight_shapes = {"weights": (n_qlayers, n_qubits)}

        self.cell = torch.nn.Linear(self.input_size, n_qubits, bias=True)
        torch.nn.init.xavier_uniform_(self.cell.weight)
        torch.nn.init.zeros_(self.cell.bias)
        
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_inputs, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }

    def forward(self, x):
        x = self.cell(x)

        for layer in range(self.n_qlayers):
            ingate = torch.sigmoid(self.VQC['forget'](x))  # forget block
            forgetgate = torch.sigmoid(self.VQC['input'](x))  # input block
            cellgate = torch.tanh(self.VQC['update'](x))  # update block
            outgate = torch.sigmoid(self.VQC['output'](x)) # output block
            
            x = torch.mul(x, forgetgate) + torch.mul(ingate, cellgate)
            x = torch.mul(outgate, torch.tanh(x))

        return x