testMipmap = function() 

	local f = {}
	local g = {}
	local h = {}
	getAttr("PASS", "PathTracer#mipmap", "Levels", 0, h);
	getAttr("PASS", "PathTracer#mipmap", "CurrentLevel",0,f)
	
	if f[1] == h[1] then
		f[1] = 0
		setAttr("PASS", "PathTracer#mipmap", "CurrentLevel",0,f)
		return false
	else
		f[1] = f[1] + 1
		setAttr("PASS", "PathTracer#mipmap", "CurrentLevel",0,f)

		k = h[1] - f[1];
		g[1] = math.pow(2,k);
		setAttr("PASS", "PathTracer#mipmap", "DIM_X", 0, g);
		setAttr("PASS", "PathTracer#mipmap", "DIM_Y", 0, g);
		
		setAttr("IMAGE_TEXTURE", "Optix Ray Tracer Render Target::mipmap", "LEVEL", 1, f);
		return true
	end
end