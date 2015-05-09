test = function() 

	local f = {}
	local g = {}
	getAttr("PASS", "voxels#mipmap", "GridCurrentLevel",0,f)
	
	if f[1] == 7 then
		f[1] = 0
		setAttr("PASS", "voxels#mipmap", "GridCurrentLevel",0,f)
		return false
	else
		f[1] = f[1] + 1
		setAttr("PASS", "voxels#mipmap", "GridCurrentLevel",0,f)

		k = 7 - f[1];
		g[1] = math.pow(2,k);
		setAttr("PASS", "voxels#mipmap", "DIM_X", 0, g);
		setAttr("PASS", "voxels#mipmap", "DIM_Y", 0, g);
		setAttr("PASS", "voxels#mipmap", "DIM_Z", 0, g);
		
		setAttr("IMAGE_TEXTURE", "Voxels::mipmap", "LEVEL", 1, f);
		return true
	end
end
