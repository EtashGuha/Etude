/*
 * @(#)MathematicaBSFFunctions.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

// BSF import switch
import org.apache.bsf.BSFManager;
//

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.jlink.MathLinkException;

/**
 * This is a utility that engine implementors may use as the Java
 * object they expose in the scripting language as "bsf". This has
 * essentially a subset of the methods in BSFManager plus some
 * stuff from the utils.
 */
public class MathematicaBSFFunctions {

  protected BSFManager manager;
  protected MathematicaBSFManager mathManager;
  
  public MathematicaBSFFunctions(BSFManager manager) {
    this.manager = manager;
    if (manager instanceof MathematicaBSFManager)
      this.mathManager = (MathematicaBSFManager)manager;
    }

  public MathematicaBSFEngine getEngine() {
    if (mathManager != null) 
      return mathManager.getMathematicaEngine();
    return null;
    }
  
  public void abort() {
    if (mathManager == null || mathManager.getMathematicaEngine() == null) return;
    try {
      mathManager.getMathematicaEngine().requestAbort();
      }
    catch (MathLinkException me) {}
    }
  
  public boolean isRunningModal() {
    if (mathManager != null) return mathManager.isRunningModal();
    else return false;
    }
    
  public Object lookupBean(String s) {
    return lookupBean(s, MathematicaObjectRegistry.SCOPE_FIRST, MathematicaObjectRegistry.SCOPE_LAST);
    }
  
  public Object lookupBean(String s, int maxScope) {
    return lookupBean(s, MathematicaObjectRegistry.SCOPE_FIRST, maxScope);
    }
   
  public Object lookupBean(String s, int minScope, int maxScope) {
    if (mathManager != null) return mathManager.lookupBean(s, minScope, maxScope);
    return manager.lookupBean(s);
    }

	public void handleException(Exception e) {
    if (mathManager == null || mathManager.getMathematicaEngine() == null) return;
	  try {
		  mathManager.getMathematicaEngine().requestHandleException(e);
			}
		catch (MathLinkException me) {}
		}
	
  public void registerBean(String beanName, Object bean) {
    registerBean(beanName, bean, MathematicaObjectRegistry.SCOPE_DEFAULT);
    }

  public void registerBean(String name, Object bean, int scope) {
    if (bean == null) unregisterBean(name);
    else {
      if (mathManager != null) mathManager.registerBean(name, bean, scope);
      else manager.registerBean(name, bean);
      }
    }

  public void unregisterBean(String beanName) {
    unregisterBean(beanName, MathematicaObjectRegistry.SCOPE_DEFAULT);
    }
    
  public void unregisterBean(String name, int scope) {
    if (mathManager != null) mathManager.unregisterBean(name, scope);
    else manager.unregisterBean(name);
    }
    
}
