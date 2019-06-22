/*
 * @(#)ApplicationDriver.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.app;

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.bsf.util.MathematicaObjectRegistry;
import com.wolfram.guikit.GUIKitDriver;

/**
 * ApplicationDriver
 *
 * Will it ever make sense for this to support/use the modal API, probably not
 */
public class ApplicationDriver extends GUIKitDriver {

  private GUIKitApplication app = null;

  public ApplicationDriver(GUIKitApplication app) {
    super();
    this.app = app;
    }

	public Object runFile(String file) {
		// We need to createEnvironment now if we want to register objects
		// for MathematicaBSFEngine to find when runFile is called
		createEnvironment();
		return super.runFile(file);
		}
	
  protected void initEnvironment() {
    registerObject(MathematicaBSFEngine.ID_USESINGLEKERNELLINK, Boolean.TRUE, MathematicaObjectRegistry.SCOPE_OBJECT);
    String customCommandLine = null;
    customCommandLine = System.getProperty("kernelLinkCommandLine", null);
    if (customCommandLine != null) {
    	setLinkCommandLine(customCommandLine);
      }
    super.initEnvironment();
    }
  
  public void destroyEnvironment(Exception e) {
    super.destroyEnvironment(e);
    if (e != null) {
      e.printStackTrace();
      }
    app.driverFinished();
    app = null;
    }

  protected void handleException(Exception e, MathematicaBSFEngine engine) {
    // By default we will assume the driver/engine is running within Mathematica
    // and only return exceptions as messages
    // Subclasses should override this method if they want to throw the exception in Java
    e.printStackTrace();
    }

}
