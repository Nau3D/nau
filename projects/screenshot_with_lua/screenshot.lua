

screenshot_test = function() 

    local f = {}

	getAttr("RENDERER", "CURRENT", "flag", 0, f)

    if f[1] == 1 then
        screenshot()
    end

    
end

