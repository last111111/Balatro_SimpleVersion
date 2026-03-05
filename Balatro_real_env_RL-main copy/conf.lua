_RELEASE_MODE = false  -- Changed to false to enable console for debugging
_DEMO = false

function love.conf(t)
	t.console = not _RELEASE_MODE
	t.title = 'Balatro'
	t.window.width = 960   -- Half of default width (1920 / 2)
    t.window.height = 540  -- Half of default height (1080 / 2)
	t.window.minwidth = 100
	t.window.minheight = 100
	t.window.resizable = true  -- Allow resizing if needed
end 
