print("teste")

syncCamera = function() 
	local f = {0,0,250,1}
	local g = {}
	local h = {}
	--ou
	-- local f = {}


	get("CAMERA", "Master", "POSITION",0,f)
	set("CAMERA", "Slave", "POSITION",0, f) 
	get("CAMERA", "Master", "VIEW",0,g)	
	set("CAMERA", "Slave", "VIEW",0, g) 
end
