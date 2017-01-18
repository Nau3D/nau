createWorkGroups3 = function ()
	local a = "PCFShadowMapping::atomicos"
	local b = "PCFShadowMapping::workGroups"
	local duvida = 652
	local atomico = 0
	--local atomicoY = 4
	
	local bf = {}
	getBuffer(a, duvida, "UINT", bf)  --Caminho Completo do Buffer, indice, tipo de var, variavel a guardar 

	local raiz = math.sqrt(bf[1])
	local valor = {}
	valor[1] = math.ceil(raiz)
	--local valorY = math.ceil(raiz)
	
	setBuffer(b, atomico, "UINT", valor)
	--setBuffer(b, atomicoY, "UINT", valor)
	
end

createWorkGroups2 = function ()
	local a = "PrimeShadowMapping2::atomicos"
	local b = "PrimeShadowMapping2::workGroups"
	local duvida = 476
	local atomico = 0
	--local atomicoY = 4
	
	local bf = {}
	getBuffer(a, duvida, "UINT", bf)  --Caminho Completo do Buffer, indice, tipo de var, variavel a guardar 

	local raiz = math.sqrt(bf[1])
	local valor = {}
	valor[1] = math.ceil(raiz)
	--local valorY = math.ceil(raiz)
	
	setBuffer(b, atomico, "UINT", valor)
	--setBuffer(b, atomicoY, "UINT", valor)
	
end

createWorkGroups = function ()
	local a = "PrimeShadowMapping::atomicos"
	local b = "PrimeShadowMapping::workGroups"
	local duvida = 276
	local atomico = 0
	--local atomicoY = 4
	
	local bf = {}
	getBuffer(a, duvida, "UINT", bf)  --Caminho Completo do Buffer, indice, tipo de var, variavel a guardar 

	local raiz = math.sqrt(bf[1])
	local valor = {}
	valor[1] = math.ceil(raiz)
	--local valorY = math.ceil(raiz)
	
	setBuffer(b, atomico, "UINT", valor)
	--setBuffer(b, atomicoY, "UINT", valor)
	
end