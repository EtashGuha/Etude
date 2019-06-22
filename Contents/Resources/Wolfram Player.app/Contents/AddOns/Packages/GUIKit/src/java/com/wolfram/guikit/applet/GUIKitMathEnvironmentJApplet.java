package com.wolfram.guikit.applet;

import com.wolfram.mathenv.mathematica.base.tool.*;
import com.wolfram.mathenv.tool.*;
import com.wolfram.mathenv.MathEnvironment;


/**
 * GUIKitMathEnvironmentJApplet allows GUIKit definitions to be run as MathEnvironment applets and
 * to use the parameterUpdate methods.
 * 
 * However, the GUIKit / MathEnvironment integration
 * isn't complete, because GUIKit still starts up its own kernel, not the FE's kernel.
 * There is rudimentary communication through the tool, though.  We get parameterUpdates
 * and the evaluate() methods.
 * 
 * @author $Author: jeffa $
 * @version $Revision: 1.2 $
 *
 */
public class GUIKitMathEnvironmentJApplet extends GUIKitJApplet implements MathToolSource{
	
    private static final long serialVersionUID = -1287989275956789948L;
  
	private AppletCellTool tool;
	
	protected void initDriver(){
		super.initDriver();
		
		MathTool myTool = MathEnvironment.sharedInstance().getTool(getParameter(MathTool.MATHTOOL_IDKEY),this);
		if(myTool != null && myTool instanceof AppletCellTool){
			tool = (AppletCellTool) myTool;
			driver.registerObject("MathEnvironmentTool",tool);
		}
		else
			tool = null;


	}

}
