printCSV = function (bufferName, outNumber, pipeline, start, size, offset)
	local maxSizes = 3
	
	if (outNumber > maxSizes) then
		return false
	else
		local file=io.open(string.format("1024-%f.csv", outNumber), "a")
		--Meter aqui um ciclo [1]->[max size of atomicos]
		local bf = {}
		--getBuffer(bufferName, 0, "UINT",bf)
		for i=start, size, offset do
			getBuffer(bufferName, i, "UINT", bf)  --Caminho Completo do Buffer, indice, tipo de var, variavel a guardar 
			file:write(string.format("%f \n", bf[1]))
		end
		file:write(string.format("end %s\n", pipeline))
		file:close()
		return true
	end
end

changeSizeShadowMap = function (s, counter)

	-- posso mudar viewport mas não posso mudar targets
	-- RUN_ONCE E CLICALR NA TECLA b FAZ RESET AO FRAME COUNT.
	local maxSizes = 3 
	local f = {}
	local g = {}
	local h = {}	
	local hF = {}	

	getAttr("PASS", s, counter, 0, f)
	getAttr("VIEWPORT", "Test2", "SIZE", 0, g)
	getAttr("PASS", s, "GridSize", 0, h)
	getAttr("PASS", s, "GridSizeF", 0, hF)
	
	if f[1] > maxSizes then
		--g[1] = 512
		--g[2] = 512
		--h[1] = 512
		--hF[1] = 512.0
		--setAttr("VIEWPORT", "Test2", "SIZE", 0, g)
		--setAttr("PASS", "textureArrays#fillTextureArray", "GridSize",0, h)
		--setAttr("PASS", "textureArrays#fillTextureArray", "GridSizeF",0, hF)
		return false
	else		
		-- Preparar o seguinte ShadowMap
		local a = 1
		
		if f[1] == 0 then --512->1024
			a = 1024
		end
		if f[1] == 1 then --1024->2048
			a = 2048 	
		end
		if f[1] == 2 then --2048->4096
			a = 4096 --width
		end
		if f[1] == 3 then
			a = 4096
		end
		
		g[1] = a
		g[2] = a
		h[1] = a
		hF[1] = a 
		
		f[1] = f[1] + 1
		setAttr("PASS", s, "ShadowMapSize",0,f)
		
		setAttr("VIEWPORT", "Test2", "SIZE", 0, g)
		setAttr("PASS", s, "GridSize",0, h)
		setAttr("PASS", s, "GridSizeF",0, hF)
		return true
	end
end

--Este não usa o changeSize xD
shadowsPrime = function ()
	local pipeline = "PrimeShadows"
	local s = "shadowsPrime#GBuffer"
	local counter = "ShadowMapSize"
	
	local b = "PrimeShadows::atomicsPrime"
	local start = 0
	local size = 16
    local offset = 4
	
	--Print Atomicos
	local maxSizes = 3
	local f = {}
	getAttr("PASS", s, counter, 0, f)
	
	if (f[1] > maxSizes) then
		return false
	else
		printCSV(b, f[1], pipeline, start, size, offset)
		f[1] = f[1] + 1
		setAttr("PASS", s, "ShadowMapSize",0,f)
		return false
	end
end


shadowsNormal = function () 
	local pipeline = "shadowsNormal"
	local s = "shadowsNormal#fillTextureArray"
	local counter = "ShadowMapSize"
	
	local b = "normalShadows::atomicsNormal"
	local start = 24
	local size = 56
    local offset = 4
	
	--Print Atomicos
	local f = {}
	getAttr("PASS", s, counter, 0, f)
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Change ShadowMaps
	return changeSizeShadowMap(s, counter)
end

shadowsExpanded = function ()
	local pipeline = "shadowsExpanded"
	local s = "shadowsAdjacency#fillTextureArray"
	local counter = "ShadowMapSize"
	
	local b = "adjacencyShadows::atomicsAdjacency"
	local start = 120
	local size = 152
    local offset = 4
	
		--Print Atomicos
	local f = {}
	getAttr("PASS", s, counter, 0, f)
	
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Change ShadowMaps
	return false
end

shadowsShrunkenAdj = function ()
	local pipeline = "shadowsShrunkenAdj"
	local s = "shadowsAdjacency#fillTextureArray"
	local counter = "ShadowMapSize"
	
	local b = "adjacencyShadows::atomicsAdjacency"
	local start = 80
	local size = 112
    local offset = 4
	
		--Print Atomicos
	local f = {}
	getAttr("PASS", s, counter, 0, f)
	
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Change ShadowMaps
	return changeSizeShadowMap(s, counter)
end

shadowsShrunken = function () 
	local pipeline = "shadowsShrunken"
	local s = "shadowsMultiple#fillTextureArray"
	local counter = "ShadowMapSize"
	
	local b = "multipleShadows::atomics"
	local start = 180
	local size = 212
    local offset = 4
	
		--Print Atomicos
	local f = {}
	getAttr("PASS", s, counter, 0, f)
	
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Change ShadowMaps
	return changeSizeShadowMap(s, counter)
end

shadowsAdjacencyCompute = function () 
	local pipeline = "shadowsAdjacencyCompute"
	local s = "SMPrimePipeline#passDepthCalc"
	local counter = "ShadowMapSize"
	
	local b = "PrimeShadowMapping::atomicos"
	local start = 280
	local size = 320
    local offset = 4
	
	--Print Atomicos Principais
	local f = {}
	getAttr("PASS", s, counter, 0, f)
	
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Print Atomicos Auxiliares
	start = 240
	size = 276
	pipeline = "CompareCA"
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Change ShadowMaps
	return changeSizeShadowMap(s, counter)
end

shadowsMultipleCompute = function () 
	local pipeline = "shadowsMultipleCompute"
	local s = "SMPrimePipelineStefan#passDepthCalc"
	local counter = "ShadowMapSize"
	
	local b = "PrimeShadowMapping2::atomicos"
	local start = 480
	local size = 520
    local offset = 4
	
	--Print Atomicos Principais
	local f = {}
	getAttr("PASS", s, counter, 0, f)
	
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Print Atomicos Auxiliares
	start = 440
	size = 476
	pipeline = "CompareSA"
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Change ShadowMaps
	return changeSizeShadowMap(s, counter)
end

shadowsPCFCompute = function () 
	local pipeline = "shadowsPCFCompute"
	local s = "SMPrimePipelinePCF#passDepthCalc"
	local counter = "ShadowMapSize"
	
	local b = "PCFShadowMapping::atomicos"
	local start = 656
	local size = 696
    local offset = 4
	
	--Print Atomicos Principais
	local f = {}
	getAttr("PASS", s, counter, 0, f)
	
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Print Atomicos Auxiliares
	start = 640
	size = 652
	pipeline = "PCFSM"
	printCSV(b, f[1], pipeline, start, size, offset)
	
	--Change ShadowMaps
	return changeSizeShadowMap(s, counter)
end