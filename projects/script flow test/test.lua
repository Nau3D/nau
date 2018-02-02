print("teste")

enterPipI = function()

	local file=io.open("flow", "a")
	file:write("Enter pipeline I\n");
	file:close()
end


execPassI = function()

	local file=io.open("flow", "a")
	file:write("Enter Pass I\n");
	file:close()
end


donePassI = function()

	local file=io.open("flow", "a")
	file:write("Done with Pass I\n");
	file:close()
end


execPassII = function()

	local file=io.open("flow", "a")
	file:write("Enter Pass II\n");
	file:close()
end


endPipI = function()

	local file=io.open("flow", "a")
	file:write("End pipeline I\n");
	file:close()
end


endPipII = function()

	local file=io.open("flow", "a")
	file:write("End pipeline II\n");
	file:close()
end


endPipIII = function()

	local file=io.open("flow", "a")
	file:write("End pipeline III\n");
	file:close()
end


testPipII = function()

	local value = {}
	getAttr("RENDERER", "CURRENT", "testVarPip2", 0, value)
	value[1] = value[1] - 1
	setAttr("RENDERER", "CURRENT", "testVarPip2", 0, value)
	local file=io.open("flow", "a")
	if value[1] == 0 then
		file:write("Test pipeline II failed\n");
		file:close()
		return false
	else 
		file:write("Test pipeline II succeeded\n");
		file:close()
		return true
	end
end


testPipIII = function()

	local value = {}
	getAttr("RENDERER", "CURRENT", "testVarPip3", 0, value)
	value[1] = value[1] - 1
	setAttr("RENDERER", "CURRENT", "testVarPip3", 0, value)
	local file=io.open("flow", "a")
	if value[1] == 0 then
		file:write("Test pipeline III failed\n");
		file:close()
		return false
	else 
		file:write("Test pipeline III succeeded\n");
		file:close()
		return true
	end
end


