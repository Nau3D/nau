testCams = function() 
	local f = {0,0,250,1}
	local f1= {0,0,0,0}
	local g = {0,0,0,0}
	local g1 = {0,0,0,0}
	local k = {0}

	getAttr("CAMERA", "MainCamera", "POSITION",0,f)
	getAttr("CAMERA", "Slave", "POSITION", 0, f1)
	getAttr("CAMERA", "MainCamera", "VIEW",0,g)	
	getAttr("CAMERA", "Slave", "VIEW",0, g1) 
	if (f[1] ~= f1[1] or f[2] ~= f1[2] or f[3] ~= f1[3] or
		g[1] ~= g1[1] or g[2] ~= g1[2] or g[3] ~= g1[3]) then

		setAttr("CURRENT", "RENDERER", "FRAME_COUNT", 0, k)
	end	
	setAttr("CAMERA", "Slave", "POSITION", 0, f) 
	setAttr("CAMERA", "Slave", "VIEW", 0, g) 

	getAttr("CURRENT", "RENDERER", "FRAME_COUNT", 0, k)
	if (k[1] > 0 and k[1] % 10 == 0) then
--	if (k[1] == 10) then
		saveTexture("Optix Ray Tracer Render Target::offscreenrenderGL")
	end	
	setBuffer("Optix Ray Tracer Render Target::test",4,"UINT",k);
end

