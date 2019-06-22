/*
 * @(#)MathematicaSecuritySupport.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.engines.javascript;

import com.wolfram.bsf.engines.MathematicaBSFEngine;

import com.wolfram.jlink.JLinkClassLoader;

import org.mozilla.javascript.SecuritySupport;

/**
 * MathematicaSecuritySupport implementation of the Rhino <tt>SecuritySupport</tt> interface is
 * meant for use within the context of Mathematica/JLink with SecuredJavaScriptEngine only.
 */
public class MathematicaSecuritySupport implements SecuritySupport {
	
	private MathematicaBSFEngine engine;
  /**
   * Default constructor
   */
  public MathematicaSecuritySupport(MathematicaBSFEngine engine){
  	this.engine = engine;
    }

  /**
   * Define and load a Java class
   */
  public Class defineClass(String name, byte[] data, Object securityDomain){
    if (securityDomain instanceof JLinkClassLoader) {
      return engine.classFromBytes(name, data);
      }
    else {
      return null;
      }
    }

  /**
   * Get the current class Context.
   */
  public Class[] getClassContext(){
    return null;
    }

  /**
   * Return the security context associated with the
   * given class.
   * In this implementation, we return the <tt>ClassLoader</tt>
   * which created the input class.
   */
  public Object getSecurityDomain(Class cl){
    return cl.getClassLoader();
    }

  /**
   * Return true if the Java class with the given name should
   * be exposed to scripts.
   *
   * In this implementation, this always return true, as
   * security is enforced by the SecurityManager's policy
   * and the Permissions granted by the URLClassLoader
   * used to load classes.
   */
  public boolean visibleToScripts(String fullClassName){
    return true;
    }

}
