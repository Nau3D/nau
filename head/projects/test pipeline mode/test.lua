startI = function()

	local f= {}
	getAttr("RENDERER", "CURRENT", "TIMER", 0, f)
	local file = io.open("test", "a");
	file:write(string.format("Start time for Pip I: %f\n", f[1]))
	file:close()
end

finish = function()
	local f= {}
	getAttr("RENDERER", "CURRENT", "TIMER", 0, f)
	local file = io.open("test", "a");
	file:write(string.format("End time: %f\n", f[1]))
	file:close()

end

startII = function()

	local f= {}
	getAttr("RENDERER", "CURRENT", "TIMER", 0, f)
	local file = io.open("test", "a");
	file:write(string.format("Start time for Pip II: %f\n", f[1]))
	file:close()
end


