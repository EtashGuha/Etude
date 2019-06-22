/*
 * @(#)MathematicaObjectRegistry.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

// BSF import switch
import org.apache.bsf.util.ObjectRegistry;
//

import com.wolfram.bsf.engines.MathematicaBSFEngine;

import java.util.Hashtable;
import java.util.Iterator;
import java.util.Vector;

/**
 * MathematicaObjectRegistry
 */
public class MathematicaObjectRegistry extends ObjectRegistry {
  
  Hashtable reg = new Hashtable();
  ObjectRegistry parent = null;

  protected int scope = SCOPE_DEFAULT;
  
  // TODO decide how Mathematica functions will specify specific scoping
  //  Choices include within the name or as a list:
  //    "PERSISTENT.NotebookSearch-windowWidth" or
  //    {"PERSISTENT", "NotebookSearch-windowWidth"}
  
  /** PERSISTENT scope should be created on framework load and
   *  hydrate from a user's filesystem serialized data store.
   * It should be the parent object registry of all runtime registries.
   * On shutdown or when requested, the peristent user's filesystem data store
   * should be updated based on all serialized objects in this runtime registry */
  public static final int SCOPE_PERSISTENT = 100000;
  
  /** SESSION scope should be a singleton registry per kernel session and should be
   * the parent registry of all OBJECT registries during the kernel session.
   * It will not get persisted but should be active during the entire kernel session */
  public static final int SCOPE_SESSION = 10000;
  
  /** OBJECT scope is an object registry created per individual Object definition and is
   * the default scoping setting for normal register/unregister requests */
  public static final int SCOPE_OBJECT = 1000;
  
  /** WIDGET scope will setup a registry on a per component level as opposed to
   * the shared OBJECT registry for all components within one definition. It is 
   * unclear whether this scope is actually needed or useful with the current API */
  public static final int SCOPE_WIDGET = 100;
  
  /** ACTION scope is a temporary registry created whenever a bind event or action is performed
   * and objects such as "#" or event argument objects are registered only while any scripts
   * are performed during the callbacks for eventing */
  public static final int SCOPE_ACTION = 10;

  public static final int SCOPE_DEFAULT = SCOPE_WIDGET;
  public static final int SCOPE_FIRST = SCOPE_ACTION;
  public static final int SCOPE_LAST = SCOPE_PERSISTENT;
  
  public static final int SCOPE_UNKNOWN = -1;
  
  public MathematicaObjectRegistry() {
    this(null, SCOPE_DEFAULT);
    }

  public MathematicaObjectRegistry(ObjectRegistry parent) {
    this(parent, 
      (parent != null && parent instanceof MathematicaObjectRegistry) ? 
         ((MathematicaObjectRegistry)parent).getScope() : SCOPE_DEFAULT);
    }

  public MathematicaObjectRegistry(ObjectRegistry parent, int scope) {
    super(parent);
    this.parent = parent;
    this.scope = scope;
    }
    
  public ObjectRegistry getParent() {return parent;}
  
  public int getScope() {return scope;}
  
  public void destroy() {
    this.parent = null;
    reg.clear();
    }
  
  // register an object
  public void register(String name, Object obj) {
    register(name, obj, SCOPE_DEFAULT);
    }

  // register an object
  public void register(String name, Object obj, int scope) {
    // Consider whether we will ever want to fail to register when this scope 
    // is already above requested scope??
    if (scope <= getScope()) {
      reg.put(name, obj);
      }
    else if (parent != null) {
      if (!(parent instanceof MathematicaObjectRegistry)) ((ObjectRegistry)parent).register(name, obj);
      else ((MathematicaObjectRegistry)parent).register(name, obj, scope);
      }
    }
  
   // unregister an object (silent if unknown name)
  public void unregister(String name) {
    unregister(name, SCOPE_DEFAULT);
    }
  
   // unregister an object (silent if unknown name)
  public void unregister(String name, int scope) {
    Object found = null;
    // unregister's scope is the maxScope looked in to provide
    // a ceiling on removal. Is this the only useful unregister??
    if (getScope() > scope) return;
    found = reg.remove(name);
    if (found == null && parent != null) {
      if (!(parent instanceof MathematicaObjectRegistry)) ((ObjectRegistry)parent).unregister(name);
      else ((MathematicaObjectRegistry)parent).unregister(name, scope);
      }
    }
  
  
  // lookup an object: cascade up if needed without exception
  
  public Object lookupObject(String name) {  
    return lookupObject(name, SCOPE_FIRST, SCOPE_LAST);
    }
    
  public Object lookupObject(String name, int minScope, int maxScope) {
    Object obj = null;
    if (getScope() <= maxScope) {
      if (getScope() >= minScope)
        obj = reg.get(name);
      if (obj == null && parent != null) {
        if (parent instanceof MathematicaObjectRegistry)
           return ((MathematicaObjectRegistry)parent).lookupObject(name, minScope, maxScope);
        else {
          try {
            obj = parent.lookup(name);
            }
          catch (IllegalArgumentException ie) {
          	return null;
          	}
          }
        }
      }
    
    return obj;
    }
    
  // lookup an object: cascade up if needed
  
  public Object lookup(String name) throws IllegalArgumentException {  
    return lookup(name, SCOPE_FIRST, SCOPE_LAST);
    }
    
  public Object lookup(String name, int minScope, int maxScope) throws IllegalArgumentException {
    Object obj = null;
    if (getScope() <= maxScope) {
      if (getScope() >= minScope)
        obj = reg.get(name);
      if (obj == null && parent != null) {
        if (parent instanceof MathematicaObjectRegistry)
           obj = ((MathematicaObjectRegistry)parent).lookupObject(name, minScope, maxScope);
        else obj = parent.lookup(name);
        }
      }
    
    if (obj == null) throw new IllegalArgumentException ("object '" + name + "' not in registry");
    else return obj;
    }

  // TODO see what the default scoping and combining of parent
  // registry keys needs to do for browsing different scoping object names
  // We may need to add a set of optional min/max scopes along with 
  // a Union and sorting into a Set
  
  public Vector getRegistryKeys(boolean filtered) {
		Vector vec = new Vector();
    if (filtered) {
      Iterator it = reg.keySet().iterator();
      while(it.hasNext()){
        String s = (String)it.next();
        if (s.startsWith(MathematicaBSFEngine.ID_PRIVATE_PREFIX) || 
            s.equals(MathematicaBSFEngine.ID_SCRIPTEVALUATOR)) continue;
        vec.add(s);
        }
      }
    else {
    	vec.addAll(reg.keySet());
      }
    if (parent != null && parent instanceof MathematicaObjectRegistry)
    	vec.addAll( ((MathematicaObjectRegistry)parent).getRegistryKeys(filtered));
    return vec;
    }

}
