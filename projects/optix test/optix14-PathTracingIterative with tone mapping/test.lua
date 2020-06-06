
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