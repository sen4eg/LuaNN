ACTIVATION_RESPONSE = 1




NeuralNetwork = {

  transfer = function( x) return 1 / (1 + math.exp(-x / ACTIVATION_RESPONSE)) end --This is the Transfer function (in this case a sigmoid)

}



function NeuralNetwork.create( _numInputs, _numOutputs, _numHiddenLayers, _neuronsPerLayer, _learningRate)

  _numInputs = _numInputs or 1

  _numOutputs = _numOutputs or 1

  _numHiddenLayers = _numHiddenLayers or math.ceil(_numInputs/2)

  _neuronsPerLayer = _neuronsPerLayer or math.ceil(_numInputs*.66666+_numOutputs)

  _learningRate = _learningRate or .5

  --order goes network[layer][neuron][wieght]

  local network = setmetatable({

    learningRate = _learningRate

  },{ __index = NeuralNetwork});

  network[1] = {}   --Input Layer

  for i = 1,_numInputs do

    network[1][i] = {}

  end

  for i = 2,_numHiddenLayers+2 do --plus 2 represents the output layer (also need to skip input layer)

    network[i] = {}

    local neuronsInLayer = _neuronsPerLayer

    if i == _numHiddenLayers+2 then

      neuronsInLayer = _numOutputs

    end

    for j = 1,neuronsInLayer do

      network[i][j] = {bias = math.random()*2-1}

      local numNeuronInputs = #network[i-1]

      for k = 1,numNeuronInputs do

        network[i][j][k] = math.random()*2-1  --return random number between -1 and 1

      end

    end

  end

  return network

end



function NeuralNetwork:forewardPropagate(...)
   local arg = {...}

  if #arg ~= #self[1] and type(arg[1]) ~= "table" then

    error("Neural Network received "..#arg.." input[s] (expected "..#self[1].." input[s])",2)

  elseif type(arg[1]) == "table" and #arg[1] ~= #self[1] then

    error("Neural Network received "..#arg[1].." input[s] (expected "..#self[1].." input[s])",2)

  end

  local outputs = {}

  for i = 1,#self do

    for j = 1,#self[i] do

      if i == 1 then

        if type(arg[1]) == "table" then

          self[i][j].result = arg[1][j]

        else

          self[i][j].result = arg[j]

        end

      else

        self[i][j].result = self[i][j].bias

        for k = 1,#self[i][j] do

          self[i][j].result = self[i][j].result + (self[i][j][k]*self[i-1][k].result)

        end

        self[i][j].result = NeuralNetwork.transfer(self[i][j].result)

        if i == #self then

          table.insert(outputs,self[i][j].result)

        end

      end

    end



  end

  return outputs

end



function NeuralNetwork:backwardPropagate(inputs,desiredOutputs)

  if #inputs ~= #self[1] then

    error("Neural Network received "..#inputs.." input[s] (expected "..#self[1].." input[s])",2)

  elseif #desiredOutputs ~= #self[#self] then

    error("Neural Network received "..#desiredOutputs.." desired output[s] (expected "..#self[#self].." desired output[s])",2)

  end

  self:forewardPropagate(inputs) --update the internal inputs and outputs

  for i = #self,2,-1 do --iterate backwards (nothing to calculate for input layer)

    local tempResults = {}

    for j = 1,#self[i] do

      if i == #self then --special calculations for output layer

        self[i][j].delta = (desiredOutputs[j] - self[i][j].result) * self[i][j].result * (1 - self[i][j].result)

      else

        local weightDelta = 0

        for k = 1,#self[i+1] do

          weightDelta = weightDelta + self[i+1][k][j]*self[i+1][k].delta

        end

        self[i][j].delta = self[i][j].result * (1 - self[i][j].result) * weightDelta

      end

    end

  end

  for i = 2,#self do

    for j = 1,#self[i] do

      self[i][j].bias = self[i][j].delta * self.learningRate

      for k = 1,#self[i][j] do

        self[i][j][k] = self[i][j][k] + self[i][j].delta * self.learningRate * self[i-1][k].result

      end

    end

  end

end



function NeuralNetwork:save()

  --[[

  File specs:

    |INFO| - should be FF BP NN

    |I| - number of inputs

    |O| - number of outputs

    |HL| - number of hidden layers

    |NHL| - number of neurons per hidden layer

    |LR| - learning rate

    |BW| - bias and weight values

  ]]--

  local data = "|INFO|FF BP NN|I|"..tostring(#self[1]).."|O|"..tostring(#self[#self]).."|HL|"..tostring(#self-2).."|NHL|"..tostring(#self[2]).."|LR|"..tostring(self.learningRate).."|BW|"

  for i = 2,#self do -- nothing to save for input layer

    for j = 1,#self[i] do

      local neuronData = tostring(self[i][j].bias).."{"

      for k = 1,#self[i][j] do

        neuronData = neuronData..tostring(self[i][j][k])

        neuronData = neuronData..","

      end

      data = data..neuronData.."}"

    end

  end

  data = data.."|END|"

  return data

end

function NeuralNetwork.load( data)

  local dataPos = string.find(data,"|")+1

  local currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)

  local dataPos = string.find(data,"|",dataPos)+1

  local _inputs, _outputs, _hiddenLayers, _neuronsPerLayer, _learningRate

  local biasWeights = {}

  local errorExit = false

  while currentChunk ~= "END" and not errorExit do

    if currentChuck == "INFO" then

      currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)

      dataPos = string.find(data,"|",dataPos)+1

      if currentChunk ~= "FF BP NN" then

        errorExit = true

      end

    elseif currentChunk == "I" then

      currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)

      dataPos = string.find(data,"|",dataPos)+1

      _inputs = tonumber(currentChunk)

    elseif currentChunk == "O" then

      currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)

      dataPos = string.find(data,"|",dataPos)+1

      _outputs = tonumber(currentChunk)

    elseif currentChunk == "HL" then

      currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)

      dataPos = string.find(data,"|",dataPos)+1

      _hiddenLayers = tonumber(currentChunk)

    elseif currentChunk == "NHL" then

      currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)

      dataPos = string.find(data,"|",dataPos)+1

      _neuronsPerLayer = tonumber(currentChunk)

    elseif currentChunk == "LR" then

      currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)

      dataPos = string.find(data,"|",dataPos)+1

      _learningRate = tonumber(currentChunk)

    elseif currentChunk == "BW" then

      currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)

      dataPos = string.find(data,"|",dataPos)+1

      local subPos = 1

      local subChunk

      for i = 1,_hiddenLayers+1 do

        biasWeights[i] = {}

        local neuronsInLayer = _neuronsPerLayer

        if i == _hiddenLayers+1 then

          neuronsInLayer = _outputs

        end

        for j = 1,neuronsInLayer do

          biasWeights[i][j] = {}

          biasWeights[i][j].bias = tonumber(string.sub(currentChunk,subPos,string.find(currentChunk,"{",subPos)-1))

          subPos = string.find(currentChunk,"{",subPos)+1

          subChunk = string.sub( currentChunk, subPos, string.find(currentChunk,",",subPos)-1)

          local maxPos = string.find(currentChunk,"}",subPos)

          while subPos < maxPos do

            table.insert(biasWeights[i][j],tonumber(subChunk))

            subPos = string.find(currentChunk,",",subPos)+1

            if string.find(currentChunk,",",subPos) ~= nil then

              subChunk = string.sub( currentChunk, subPos, string.find(currentChunk,",",subPos)-1)

            end

          end

          subPos = maxPos+1

        end

      end

    end

    currentChunk = string.sub( data, dataPos, string.find(data,"|",dataPos)-1)

    dataPos = string.find(data,"|",dataPos)+1

  end

  if errorExit then

    error("Failed to load Neural Network:"..currentChunk,2)

  end

  local network = setmetatable({

    learningRate = _learningRate

  },{ __index = NeuralNetwork});

  network[1] = {}   --Input Layer

  for i = 1,_inputs do

    network[1][i] = {}

  end

  for i = 2,_hiddenLayers+2 do --plus 2 represents the output layer (also need to skip input layer)

    network[i] = {}

    local neuronsInLayer = _neuronsPerLayer

    if i == _hiddenLayers+2 then

      neuronsInLayer = _outputs

    end

    for j = 1,neuronsInLayer do

      network[i][j] = {bias = biasWeights[i-1][j].bias}

      local numNeuronInputs = #network[i-1]

      for k = 1,numNeuronInputs do

        network[i][j][k] = biasWeights[i-1][j][k]

      end

    end

  end

  return network

end
