print("teste")

syncCamera = function() 
	local f = {0,0,250,1}
	local g = {0,0,0,0}
	local h = {}

	getAttr("CAMERA", "Master", "POSITION",0,f)
	setAttr("CAMERA", "Slave", "POSITION", 0, f) 
	getAttr("CAMERA", "Master", "VIEW",0,g)	
	setAttr("CAMERA", "Slave", "VIEW",0, g) 
end

