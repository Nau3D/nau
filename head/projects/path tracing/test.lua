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

		setAttr("RENDERER", "CURRENT", "FRAME_COUNT", 0, k)
		setAttr("CAMERA", "Slave", "POSITION", 0, f) 
		setAttr("CAMERA", "Slave", "VIEW", 0, g) 
	end	

--	getAttr("RENDERER", "CURRENT", "FRAME_COUNT", 0, k)
--	if (k[1] > 0 and k[1] % 10 == 0) then
--		saveTexture("Optix Ray Tracer Render Target::offscreenrenderGL")
--	end	
--	setBuffer("Optix Ray Tracer Render Target::test",4,"UINT",k);
end

testMipmap = function() 

	local f = {0}
	local g = {0}
	local h = {0}
	local k = {0}
	getAttr("PASS", "PathTracer#mipmap", "Levels", 0, h);
	getAttr("PASS", "PathTracer#mipmap", "CurrentLevel",0,f)
	
	if f[1] == h[1] then
		f[1] = 0
		setAttr("PASS", "PathTracer#mipmap", "CurrentLevel",0,f)
		return false
	else
		f[1] = f[1] + 1
		k[1] = h[1] - f[1];
		if k[1] == 0 then
			g[1] = 1;
		else g[1] = 2;	
			for i=2,k[1] do
				g[1] = g[1] * 2
			end	
		end
		
		setAttr("PASS", "PathTracer#mipmap", "CurrentLevel",0,f)
		setAttr("PASS", "PathTracer#mipmap", "DIM_X", 0, g);
		setAttr("PASS", "PathTracer#mipmap", "DIM_Y", 0, g);
		
		setAttr("IMAGE_TEXTURE", "Optix Ray Tracer Render Target::mipmap", "LEVEL", 1, f);
		return true
	end
end