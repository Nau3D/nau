print("teste")

syncCamera = function() 
	local f = {0,0,250,1}
	local g = {}
	local h = {}


	getAttr("CAMERA", "Master", "POSITION",0,f)
	-- lua arrays start at 1
	-- f[2] = f[2] + 20
	setAttr("CAMERA", "Slave", "POSITION",0, f) 
	getAttr("CAMERA", "Master", "VIEW",0,g)	
	setAttr("CAMERA", "Slave", "VIEW",0, g) 
	getBuffer("D:\\Nau\\head\\models\\Teapot.3ds#0:normal", 0, "VEC4", h)
	local file=io.open("test", "a")
	file:write(string.format("%f %f %f\n",h[1], h[2], h[3]))
	file:close()
end

enterPassMaster = function()

	local file=io.open("flow", "a")
	file:write("Enter pass master\n");
	file:close()
end
	
enterPassSlave = function()

	local file=io.open("flow", "a")
	file:write("Enter pass slave\n");
	file:close()
end

leavingPassMaster = function()

	local file=io.open("flow", "a")
	file:write("Leaving pass master\n");
	file:close()
end

leavingPassSlave = function()

	local file=io.open("flow", "a")
	file:write("Leaving pass slave\n");
	file:close()
end
