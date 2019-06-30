/*
 * @(#)MathematicaBSFManager.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

// BSF import switch
import java.io.PrintStream;

import org.apache.bsf.BSFEngine;
import org.apache.bsf.BSFException;
import org.apache.bsf.BSFManager;
import org.apache.bsf.util.ObjectRegistry;
//

import com.wolfram.bsf.engines.MathematicaBSFEngine;

/**
 * MathematicaBSFManager
 */
public class MathematicaBSFManager extends BSFManager {
  // Even though we may be using a modified Languages properties file
  // we at least make sure we add them here too so that certain deployments can 
  // work with an unmodifed bsf.jar
  
  public static final String MATHEMATICA_LANGUAGE_NAME = "mathematica";
  
  private static final String MATHEMATICA_SCRIPTENGINE_CLASS = "com.wolfram.bsf.engines.MathematicaBSFEngine";
  private static final String[] MATHEMATICA_SCRIPTENGINE_EXTENSIONS = new String[] {"m"};

  private static final String JAVASCRIPT_SCRIPTENGINE_CLASS = "com.wolfram.bsf.engines.javascript.SecuredJavaScriptEngine";
  private static final String[] JAVASCRIPT_SCRIPTENGINE_EXTENSIONS = new String[] {"js"};

  static {
    BSFManager.registerScriptingEngine(MATHEMATICA_LANGUAGE_NAME,
      MATHEMATICA_SCRIPTENGINE_CLASS, MATHEMATICA_SCRIPTENGINE_EXTENSIONS);
    // Ok, so other engine langauge names are all lowercase but lets make this work
    BSFManager.registerScriptingEngine("Mathematica",
      MATHEMATICA_SCRIPTENGINE_CLASS, MATHEMATICA_SCRIPTENGINE_EXTENSIONS);

    // The ONLY reason we register a subclass of the javascript engine is to make
    // the javascript engine work against a js.jar on the classpath with security turned
    // on. When the default engine supports addding the securitySupport we will stop
    // registering this alternate engine.
    BSFManager.registerScriptingEngine("javascript",
      JAVASCRIPT_SCRIPTENGINE_CLASS, JAVASCRIPT_SCRIPTENGINE_EXTENSIONS);

    }
    
  private static final String PRIMITIVE_ARRAY[] = {"boolean", "byte", "char", "short", "int", "long", "float", "double"};
  private static final Class CLASS_ARRAY[] = {Boolean.TYPE, Byte.TYPE, Character.TYPE, Short.TYPE, Integer.TYPE,
        Long.TYPE, Float.TYPE, Double.TYPE};
        
  public static ClassLoader classLoader = null;
  
  private LinkInitializer linkInitializer = null;
  
  private MathematicaBSFEnvironment mathematicaBSFEnvironment = null;
  
  // debug stream
  protected PrintStream debugStream = System.err;

  // debug mode flag
  boolean debug = false;
  
  protected static ThreadLocal threadLocalObjectRegistry = new ThreadLocal();
  
  public MathematicaBSFManager() {
    this(null);
    }
    
  public MathematicaBSFManager(MathematicaBSFManager parent) {
    super();
    if (parent != null)
      setObjectRegistry(parent.getObjectRegistry());
    }
    
  public void terminate() {
    super.terminate();
    if (getObjectRegistry() != null && getObjectRegistry() instanceof MathematicaObjectRegistry)
      ((MathematicaObjectRegistry)getObjectRegistry()).destroy();
    }
    
  public boolean isRunningModal() {
    if (mathematicaBSFEnvironment != null) return mathematicaBSFEnvironment.isRunningModal();
    else return false;
    }
  
  public MathematicaBSFEnvironment getMathematicaBSFEnvironment() {return mathematicaBSFEnvironment;}
  public void setMathematicaBSFEnvironment(MathematicaBSFEnvironment env) {
    mathematicaBSFEnvironment = env;
    }
  
  //////////////////////////////////////////////////////////////////////////

  /**
   * Send debug output to this print stream.  Default is System.err.
   *
   * @param debugStream stream to direct debug output to.
   */
  public void setDebugStream (PrintStream debugStream) {
    pcs.firePropertyChange ("debugStream", this.debugStream, debugStream);
    this.debugStream = debugStream;
  }

  /**
   * Get debug stream
   */
  public PrintStream getDebugStream () {
    return debugStream;
  }

  //////////////////////////////////////////////////////////////////////////

  /**
   * Turn on off debugging output to System.err. Default is off (false).
   *
   * @param debug value to set debug flag to
   */
  public void setDebug (boolean debug) {
    pcs.firePropertyChange ("debug", new Boolean (this.debug), 
          new Boolean (debug));
    this.debug = debug;
  }

  /**
   * Get debug status
   */
  public boolean getDebug () {
    return debug;
  }
  
  public void initializeLink(String contextName) throws MathematicaBSFException {
    if (linkInitializer != null)
      linkInitializer.initializeLink(this, contextName);
    }
  
  public void setLinkInitializer(LinkInitializer li) {
    linkInitializer = li;
    }
  
  public int getScope() {
     if (getObjectRegistry() instanceof MathematicaObjectRegistry) 
      return ((MathematicaObjectRegistry)getObjectRegistry()).getScope();
     else return MathematicaObjectRegistry.SCOPE_UNKNOWN;
     }
     
  /* We override the default get/setObjectRegistry methods because
   * we want to support a ThreadLocal object registry for ACTION scope
   * evaluations on a running thread.
   */
  
  public void setObjectRegistry(ObjectRegistry objectRegistry) {
     setObjectRegistry(objectRegistry, false);
     }  
  
    /**
     * Set the object registry used by this manager. By default a new
     * one is created when the manager is new'ed and this overwrites 
     * that one.
     *
     * @param objectRegistry the registry to use
     */
  public void setObjectRegistry(ObjectRegistry objectRegistry, boolean toLocal) {
    if (objectRegistry == null) {
      threadLocalObjectRegistry.set(null);
      }
    else {
      if (false || threadLocalObjectRegistry.get() != null) {
        threadLocalObjectRegistry.set(objectRegistry);
        return;
        }
      }
    this.objectRegistry = objectRegistry;
    }
    
    /**
     * Return the current object registry of the manager.
     *
     * @return the current registry.
     */
  public ObjectRegistry getObjectRegistry() {
    if (threadLocalObjectRegistry.get() != null) return (ObjectRegistry)threadLocalObjectRegistry.get();
    return objectRegistry;
    }
    
  public MathematicaObjectRegistry pushObjectRegistry(int scope) {
    return pushObjectRegistry(scope, false);
    }
  
  public MathematicaObjectRegistry pushObjectRegistry(int scope, boolean toLocal) {
    MathematicaObjectRegistry newRegistry = new MathematicaObjectRegistry(getObjectRegistry(), scope);
    setObjectRegistry(newRegistry, toLocal);
    return newRegistry;
    }
  
  public void popObjectRegistry() {
    popObjectRegistry(false);
    }
  
  public void popObjectRegistry(boolean fromLocal) {
    ObjectRegistry current = getObjectRegistry();
    if (current instanceof MathematicaObjectRegistry && 
      ((MathematicaObjectRegistry)current).getParent() != null) {
      if (fromLocal) {
        threadLocalObjectRegistry.set(null);
        }
      setObjectRegistry(((MathematicaObjectRegistry)current).getParent(), false);
      ((MathematicaObjectRegistry)current).destroy();
      }
    }
    
  public String[] getReferenceNames(boolean filtered) {
    if (getObjectRegistry() != null && getObjectRegistry() instanceof MathematicaObjectRegistry)
      return (String[])((MathematicaObjectRegistry)getObjectRegistry()).getRegistryKeys(filtered).toArray( new String[]{});
    else return new String[]{};
    }
    
  public MathematicaBSFEngine getMathematicaEngine() {
    classLoader = getClass().getClassLoader();
    
    try {
      BSFEngine mathEngine = loadScriptingEngine(MATHEMATICA_LANGUAGE_NAME);

      if (mathEngine != null && mathEngine instanceof MathematicaBSFEngine) {
        return (MathematicaBSFEngine)mathEngine;
        }
      }
    catch (BSFException e) {
      // We may want to throw an exception here
      e.printStackTrace();
      }
    return null;
    }
    
  public boolean isMathematicaEngineLoaded() {
    MathematicaBSFEngine eng = (MathematicaBSFEngine)loadedEngines.get(MATHEMATICA_LANGUAGE_NAME);
    return eng != null;
    }
    
  public Object evaluateScript(String lang, String context, Object scriptContent) throws MathematicaBSFException {
    try {
      return super.eval(lang, context, -1, -1, scriptContent);
      }
    catch (BSFException be) {
      throw new MathematicaBSFException(be.getMessage());
      }
    }
    
  // Global requests should register beans in the environment
  // first and to the manager's registry only for local settings
  // as a manager's object registry is local and the Environment's
  // registry is shared

  public void registerBean(String beanName, Object bean) {
    registerBean(beanName, bean, MathematicaObjectRegistry.SCOPE_DEFAULT);
    }

  public void registerBean(String beanName, Object bean, int scope) {
    if (getObjectRegistry() instanceof MathematicaObjectRegistry) {
      ((MathematicaObjectRegistry)getObjectRegistry()).register(beanName, bean, scope);
      } 
    else
      super.registerBean(beanName, bean);
    }

  public void unregisterBean(String beanName) {
    unregisterBean(beanName, MathematicaObjectRegistry.SCOPE_DEFAULT);
    }
    
  public void unregisterBean(String beanName, int scope) {
    if (getObjectRegistry() instanceof MathematicaObjectRegistry) {
      ((MathematicaObjectRegistry)getObjectRegistry()).unregister(beanName, scope);
      }
    else {
      super.unregisterBean(beanName);
      }
    }

	public Object lookupBean(String s) {
		return lookupBean(s, MathematicaObjectRegistry.SCOPE_FIRST, MathematicaObjectRegistry.SCOPE_LAST);
		}
  
	public Object lookupBean(String s, int maxScope) {
		return lookupBean(s, MathematicaObjectRegistry.SCOPE_FIRST, maxScope);
		}
   
  public Object lookupBean(String s, int minScope, int maxScope) {
    Object obj = null;
    if (s == null) return null;
 
    if (obj == null) {
      // check if it is a "class:" string and return Class
      if(s.startsWith("class:")) {
        try {
          obj = resolveClassName(s.substring(6));
          }
        catch (BSFException ie) {}
        }
      }

    if(obj == null) {
			if (getObjectRegistry() instanceof MathematicaObjectRegistry) {
				obj = ((MathematicaObjectRegistry)getObjectRegistry()).lookupObject(s, minScope, maxScope);
				}
			else {
				try {
					obj = super.lookupBean(s);
					}
				catch(IllegalArgumentException ex) {}
				}
			}

    return obj;
    }
   
 
  public Class resolveClassName(String s) throws MathematicaBSFException {
      if(s == null)
        throw new MathematicaBSFException(BSFException.REASON_INVALID_ARGUMENT, "unable to resolve class '" + s + "'");
     
      for(int i = 0; i < PRIMITIVE_ARRAY.length; i++)
        if(s.equals(PRIMITIVE_ARRAY[i]))
          return CLASS_ARRAY[i];

      try {
        if(classLoader == null)
          return Class.forName(s);
        else
          return classLoader.loadClass(s);
        }
      catch(ClassNotFoundException _ex) {
        throw new MathematicaBSFException(BSFException.REASON_INVALID_ARGUMENT, "unable to resolve class '" + s + "'");
        }
      }
      
}
