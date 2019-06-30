/*
 * @(#)GUIKitEnvironment.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.tree.TreeNode;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeModel;
import javax.swing.table.TableModel;
import javax.swing.ListModel;
import javax.swing.DefaultListModel;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.table.DefaultTableModel;

import org.apache.bsf.util.type.TypeConvertor;

import diva.graph.GraphModel;

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.bsf.util.LinkInitializer;
import com.wolfram.bsf.util.MathematicaBSFEnvironment;
import com.wolfram.bsf.util.MathematicaBSFException;
import com.wolfram.bsf.util.MathematicaBSFManager;
import com.wolfram.bsf.util.MathematicaObjectRegistry;
import com.wolfram.bsf.util.type.ExprTypeConvertor;
import com.wolfram.bsf.util.type.MathematicaTypeConvertorRegistry;

import com.wolfram.guikit.graph.ExprGraphTypeConvertor;
import com.wolfram.guikit.swing.ItemListModel;
import com.wolfram.guikit.swing.ItemTableModel;

import com.wolfram.jlink.Expr;
import com.wolfram.jlink.KernelLink;
import com.wolfram.jlink.MathLinkException;
import com.wolfram.jlink.StdLink;

/**
 * GUIKitEnvironment
 */
public class GUIKitEnvironment implements Cloneable, LinkInitializer, MathematicaBSFEnvironment {

 /**
   * The version string identifying this release.
   */
  public static final String VERSION = "1.0.3";
  
  /**
   * The major version number identifying this release.
   */
  public static final double VERSION_NUMBER = 1.0;
  
  public static final String PACKAGE_CONTEXT = "GUIKit`";
  public static final String RESOLVE_FUNCTION = PACKAGE_CONTEXT + "Private`resolveMathematicaFile";
  
  private static final String BSF_SYMBOL = MathematicaBSFEngine.BSF_SYMBOL;
  private static final String WRAPPER_BASE = PACKAGE_CONTEXT + "Private`guiWrap";
  private static final String OBJECTREF_SYMBOL = "WidgetReference";

  protected Vector childrenEnvironments = null;
  protected GUIKitEnvironment parentEnvironment = null;
  protected GUIKitDriver driver = null;
  
  protected MathematicaBSFManager bsfManager = null;
  
  protected ArrayList managedEnvironmentObjects = new ArrayList();
  
  // TODO Think about whether we move out this required static
  // class loading dependency on diva 
  // and whether we can register convertors elsewhere but before they
  // may be needed
  static {
    TypeConvertor convertor = null;
    convertor = new ExprTypeConvertor();
    
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(TreeModel.class, Expr.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(TreeNode.class, Expr.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(TableModel.class, Expr.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(ListModel.class, Expr.class, convertor);

    // lookup a convertor
    // TODO look into changing how to/froms are registered
    // so that assignable classes do not need to be explicitly
    // registered. Consider after to/from and key check last check
    // would be to loop registers, match to and if from assignable from use
    // Needs to change how hashtables store converters as values and instead
    // consider a wrapper class that also groups the to and from classes in the val
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(DefaultListModel.class, Expr.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(DefaultTreeModel.class, Expr.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(DefaultTableModel.class, Expr.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(DefaultMutableTreeNode.class, Expr.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(ItemListModel.class, Expr.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(ItemTableModel.class, Expr.class, convertor);
    
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(Expr.class, TreeModel.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(Expr.class, TreeNode.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(Expr.class, TableModel.class, convertor);
    MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(Expr.class, ListModel.class, convertor);
    
		convertor = new ExprGraphTypeConvertor();
		MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(GraphModel.class, Expr.class, convertor);
		MathematicaTypeConvertorRegistry.typeConvertorRegistry.register(Expr.class, GraphModel.class, convertor);
    }
  
  public GUIKitEnvironment() {
  	this(null, MathematicaObjectRegistry.SCOPE_DEFAULT);
  	}

  public GUIKitEnvironment(GUIKitEnvironment parentEnv) {
    this(parentEnv, MathematicaObjectRegistry.SCOPE_DEFAULT);
    }
    
    
  public GUIKitEnvironment(GUIKitEnvironment parentEnv, int scope) {
    // We want a new env registry per driver instance
    GUIKitEnvironment useParentEnv = parentEnv;
    // Maybe this needs to be <= so that WIDGETS will jump to OBJECT parent??
    // Also once we have PERSISTENT and SESSION singletons at top this will
    // be guaranteed
    while(useParentEnv != null && useParentEnv.getParentEnvironment() != null && 
         useParentEnv.getBSFManager().getScope() < scope) {
      useParentEnv = useParentEnv.getParentEnvironment();
      }
    if (useParentEnv != null)
      useParentEnv.addChildEnvironment(this);
    
    bsfManager = new MathematicaBSFManager(useParentEnv != null ? useParentEnv.getBSFManager() : null);
    bsfManager.setLinkInitializer(this);
    bsfManager.setMathematicaBSFEnvironment(this);
    
    setParentEnvironment(useParentEnv);

    // Here we make sure pushed Object registry in next call is the root
    // TODO we will instead set the registry to be the PERSISTENT static parent for all new environments
		if (useParentEnv == null)
		  bsfManager.setObjectRegistry(null);
		  
    pushObjectRegistry(scope);
    
    // register this environment in the bsfmanager so it can look up
    // GUIKIT objects without importing them into bsf code
    registerObject(GUIKitDriver.ID_GUIKITENV, this);
    }

  public boolean isRunningModal() {
    if (getDriver() != null) return getDriver().getExecuteAsModal();
    else return false;
    }
    
  public static ClassLoader getClassLoader() {return MathematicaBSFManager.classLoader;}
  
  public MathematicaObjectRegistry pushObjectRegistry(int scope) {
    return pushObjectRegistry(scope, false);
    }
  public MathematicaObjectRegistry pushObjectRegistry(int scope, boolean checkLocal) {
    return bsfManager.pushObjectRegistry(scope, checkLocal);
    }
  public void popObjectRegistry() {
    popObjectRegistry(false);
    }
  public void popObjectRegistry(boolean checkLocal) {
    bsfManager.popObjectRegistry(checkLocal);
    }
  
  public boolean getDebug() {return bsfManager.getDebug();}
  public void setDebug(boolean debug) { bsfManager.setDebug(debug);}
  public PrintStream getDebugStream() {return bsfManager.getDebugStream();}
  
  protected MathematicaBSFManager getBSFManager() {return bsfManager;}
  
  public String[] getReferenceNames(boolean filtered) {
    return bsfManager.getReferenceNames(filtered);
    }
   
	public Object lookupObject(String id) {
		return lookupObject(id, MathematicaObjectRegistry.SCOPE_FIRST, MathematicaObjectRegistry.SCOPE_LAST);
		}
		
	public Object lookupObject(String id, int maxScope) {
		return lookupObject(id, MathematicaObjectRegistry.SCOPE_FIRST, maxScope);
		}
		
  public Object lookupObject(String id, int minScope, int maxScope) {
    if (id == null) return null;
    return bsfManager.lookupBean(id, minScope, maxScope);
    }

  public void registerObject(String id, Object obj) {
    registerObject(id, obj, MathematicaObjectRegistry.SCOPE_DEFAULT);
    }
  public void registerObject(String id, Object obj, int scope) {
    if (id == null) return;
    if (obj == null) unregisterObject(id, scope);
    // When talking to driver user registers should go to manager's object registry by default
    else bsfManager.registerBean(id, obj, scope);
    }

  public void unregisterObject(String id) {
    unregisterObject(id, MathematicaObjectRegistry.SCOPE_DEFAULT);
    }
  public void unregisterObject(String id, int scope) {
    if (id == null) return;
    // When talking to driver user registers should go to manager's object registry by default
    bsfManager.unregisterBean(id, scope);
    }
    
  public boolean isMathematicaEngineLoaded() {return bsfManager.isMathematicaEngineLoaded();}
  
  public MathematicaBSFEngine getMathematicaEngine() {
     return getMathematicaEngine(false);
     }
  public MathematicaBSFEngine getMathematicaEngine(boolean allowInherited) {
    if (allowInherited && getParentEnvironment() != null) {
      return getParentEnvironment().getMathematicaEngine();
      }
    return bsfManager.getMathematicaEngine();
    }
    
  public Object evaluateScript(String context, Object scriptContent) throws GUIKitException {
    return evaluateScript(MathematicaBSFManager.MATHEMATICA_LANGUAGE_NAME, context, scriptContent);
    }
    
  public Object evaluateScript(String lang, String context, Object scriptContent) throws GUIKitException {
    Object result = null;
    try {
      result = bsfManager.evaluateScript(lang, context, scriptContent);
      }
    catch (MathematicaBSFException be) {
      throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR, "exception from BSF: " + be, be);
    }
    return result;
    }
   
  public Class resolveClassName(String s) throws GUIKitException {
    Class result = null;
    try {
      result = bsfManager.resolveClassName(s);
    }
    catch (MathematicaBSFException be) {
      throw new GUIKitException(GUIKitException.REASON_UNKNOWN_CLASS, be.getMessage());
    }
    return result;
    }
  
  public GUIKitDriver getDriver() {return driver;}
  public void setDriver(GUIKitDriver d) {driver = d;}
  
  public GUIKitEnvironment getParentEnvironment() {return parentEnvironment;}
  
  protected void setParentEnvironment(GUIKitEnvironment env) {
    // TODO we may need to check if registered beans in parent might
    // be adversely found in this children and we need to 
    // 'block' certain settings??
    parentEnvironment = env;
    if (parentEnvironment != null) {
      setDriver(parentEnvironment.getDriver());
      setDebug(parentEnvironment.getDebug());
      }
    }
    
  protected void addChildEnvironment(GUIKitEnvironment env) {
    if (childrenEnvironments == null) {
      childrenEnvironments = new Vector(2);
      }
    childrenEnvironments.add(env);
    }
    
  public void destroyChildrenEnvironments() {
		if (childrenEnvironments != null) {
			Iterator it = childrenEnvironments.iterator();
			while(it.hasNext()) {
				((GUIKitEnvironment)it.next()).destroy();
				}
			}
		// TODO we may want to zero and null out environment state
		if (childrenEnvironments != null)
			childrenEnvironments.clear();
  	}
  
  public void addGUIKitEnvironmentObject(GUIKitEnvironmentObject o) {
    managedEnvironmentObjects.add(o);
    }
    
  public void destroy() {
		destroyChildrenEnvironments();
    bsfManager.terminate();
    Iterator it = managedEnvironmentObjects.iterator();
    while (it.hasNext()) {
      Object o = it.next();
      if (o != null && o instanceof GUIKitEnvironmentObject) {
        ((GUIKitEnvironmentObject)o).setGUIKitEnvironment(null);
        }
      }
    driver = null;
    }

  private void appendGUIKitFunction(StringBuffer mathCode, String contextName, String name) {
    mathCode.append(contextName);
    mathCode.append(name);
    mathCode.append("[args___] := Module[{result}, result = "); 
    mathCode.append(WRAPPER_BASE);
    mathCode.append(name);
    mathCode.append("[");
    mathCode.append(contextName);
    mathCode.append(BSF_SYMBOL);
    mathCode.append(", args]; result");
    mathCode.append(" /; Head[result] =!= ");
    mathCode.append(WRAPPER_BASE);
    mathCode.append(name); 
    mathCode.append("];");
    mathCode.append('\n');
    }
  
  private void appendGUIKitResolveFunction(StringBuffer mathCode, String contextName, String name) {
    mathCode.append("Attributes[" + contextName + name + "] = {HoldRest};");
    mathCode.append('\n');
    mathCode.append(contextName + name + "[args__] := ");
    mathCode.append(PACKAGE_CONTEXT + "GUIResolve[ ");
    mathCode.append(PACKAGE_CONTEXT + "Private`ConvertResolveContent[" +
      PACKAGE_CONTEXT + name + ", {args}, \"" + contextName + "\"]");
    // If we turn this off then a Script based Widget will create its own
    // new script context. This actually might be the desired result and
    // not share context or functions of Script, maybe not.
    // If we turn this off we would also need to resolve the checks for this symbol in
    // the GUIResolve package functions
    mathCode.append(", " + PACKAGE_CONTEXT + "GUIObject -> " + contextName + MathematicaBSFEngine.DRIVER_SYMBOL);
    mathCode.append("];");
    mathCode.append('\n');
    }
    
  public void initializeLink(MathematicaBSFManager mathMgr, String contextName) 
  	  throws MathematicaBSFException {
      KernelLink ml = StdLink.getLink();
      
    try {
      
    StdLink.requestTransaction();
    synchronized (ml) {
      ml.evaluate("Needs[\"" + PACKAGE_CONTEXT + "\"]");
      ml.discardAnswer();
      }
        
      StdLink.requestTransaction();
      synchronized (ml) {
        ml.evaluate( "Begin[\"" + contextName + "\"]");
        ml.discardAnswer();
        }

      // We declare these Mathematica rules here so that all kernel instances
      // can use the BSF functionality without requiring a separate Mathematica package
      // installation, also the PACKAGE_CONTEXT package working is optional and will
      // not affect the basic BSF engine

      StringBuffer mathCode = new StringBuffer();

      mathCode.append('(');
      mathCode.append('\n');

      // PACKAGE_CONTEXT WidgetReference versions call to PACKAGE_CONTEXT functions
      // when needed, currently they do nothing different
      
      mathCode.append(PACKAGE_CONTEXT + "Private`initGUI[];");
        
      mathCode.append('\n');

      appendGUIKitFunction(mathCode, contextName, OBJECTREF_SYMBOL);
      appendGUIKitFunction(mathCode, contextName, "Set" + OBJECTREF_SYMBOL);
      appendGUIKitFunction(mathCode, contextName, "Unset" + OBJECTREF_SYMBOL);
      appendGUIKitFunction(mathCode, contextName, "PropertyValue");
      appendGUIKitFunction(mathCode, contextName, "SetPropertyValue");
      appendGUIKitFunction(mathCode, contextName, "InvokeMethod");
      appendGUIKitFunction(mathCode, contextName, "CloseGUIObject");
      appendGUIKitFunction(mathCode, contextName, "GUIObject");
      appendGUIKitFunction(mathCode, contextName, "GUIInformation");
      
      // without wrapper version can only use target strings and not existing JavaObjects
      appendGUIKitResolveFunction(mathCode, contextName, "BindEvent");
      
      // Making Widget expressions automatically execute in Scripts may or may not be desired by default.
      // One could possibly call GUIResolve[] if we turn this off but since many other functions
      // evaluate in Scripts it is difficult to build up a non-evaluated complete Widget[]
      // unless one keeps it Unevaluated.
      appendGUIKitResolveFunction(mathCode, contextName, "Widget");

      mathCode.append(')');

     // Send mathCode to kernel
      StdLink.requestTransaction();
      synchronized (ml) {
        ml.evaluate( mathCode.toString() );
        ml.discardAnswer();
        }

    StdLink.requestTransaction();
    synchronized (ml) {
      ml.evaluate( "End[]");
      ml.discardAnswer();
      }
      
    }
    catch (MathLinkException e) {
			// How should we cleanup/fail with initialize, probably rethrow BSFException
			if (mathMgr != null && mathMgr.getDebug())
				e.printStackTrace(mathMgr.getDebugStream());

			throw new MathematicaBSFException(MathematicaBSFException.REASON_OTHER_ERROR, 
				(e.getMessage() != null) ? e.getMessage() : e.getClass().getName(), e);
    	}
    finally {}
      
    }
 
}
