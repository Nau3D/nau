test = function() 

	local f = {}
	local g = {}
	local h = {}
	local k = {}
	getAttr("PASS", "CURRENT", "GridLevels", 0, h);
	getAttr("PASS", "CURRENT", "GridCurrentLevel",0,f)
	
	if f[1] == h[1] then
		f[1] = 0
		setAttr("PASS", "CURRENT", "GridCurrentLevel",0,f)
		return false
	else
		f[1] = f[1] + 1
		setAttr("PASS", "CURRENT", "GridCurrentLevel",0,f)

		k[1] = h[1] - f[1];
		if k[1] == 0 then
			g[1] = 1;
		else g[1] = 2;	
			for i=2,k[1] do
				g[1] = g[1] * 2
			end	
		end
		setAttr("PASS", "CURRENT", "DIM_X", 0, g);
		setAttr("PASS", "CURRENT", "DIM_Y", 0, g);
		setAttr("PASS", "CURRENT", "DIM_Z", 0, g);
		
		setAttr("IMAGE_TEXTURE", "Voxels::mipmap", "LEVEL", 1, f);
		setAttr("IMAGE_TEXTURE", "Voxels::mipmap", "LEVEL", 2, f);
		return true
	end
end


testMipmap = function() 

	local f = {0}
	local g = {0}
	local h = {0}
	local k = {0}
	getAttr("PASS", "CURRENT", "Levels", 0, h);
	getAttr("PASS", "CURRENT", "CurrentLevel",0,f)
	
	if f[1] == h[1] then
		f[1] = 0
		setAttr("PASS", "CURRENT", "CurrentLevel",0,f)
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
		
		setAttr("PASS", "CURRENT", "CurrentLevel",0,f)
		setAttr("PASS", "CURRENT", "DIM_X", 0, g);
		setAttr("PASS", "CURRENT", "DIM_Y", 0, g);
		
		setAttr("IMAGE_TEXTURE", "Voxels::mipmapLuminance", "LEVEL", 1, f);
		return true
	end
end