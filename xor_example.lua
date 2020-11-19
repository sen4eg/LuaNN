math.randomseed(os.time())
dofile("NeuralNetwork.lua")      

network = NeuralNetwork.create(2,1,2,4,0.3)

print("Training the neural network:")
attempts = 10000 -- number of times to do backpropagation
 for i = 1,attempts do
  network:backwardPropagate({0,0},{0}) 
  network:backwardPropagate({1,0},{1})
  network:backwardPropagate({0,1},{1})
  network:backwardPropagate({1,1},{0})
end

print("Results:")
print("0 0 | "..network:forewardPropagate(0,0)[1])
print("1 0 | "..network:forewardPropagate(1,0)[1])
print("0 1 | "..network:forewardPropagate(0,1)[1])
print("1 1 | "..network:forewardPropagate(1,1)[1])
