/*
 * @(#)MathematicaReflectionUtils.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

import java.beans.*;
import java.io.IOException;
import java.lang.reflect.*;

import com.wolfram.bsf.util.type.MathematicaTypeConvertorRegistry;
import com.wolfram.bsf.util.type.TypedObject;
import com.wolfram.bsf.util.type.TypedObjectFactory;

/**
 * This file is a collection of reflection utilities specific to Mathematica.
 */
public class MathematicaReflectionUtils {

  // Copied code of ReflectionUtils to get at needed change of classloader

  //////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////

  /**
   * Create a bean using given class loader and using the appropriate
   * constructor for the given args of the given arg types.

   * @param cl        the class loader to use. If null, Class.forName is used.
   * @param className name of class to instantiate
   * @param argTypes  array of argument types
   * @param args      array of arguments
   *
   * @return the newly created bean
   *
   * @exception ClassNotFoundException    if class is not loaded
   * @exception NoSuchMethodException     if constructor can't be found
   * @exception InstantiationException    if class can't be instantiated
   * @exception IllegalAccessException    if class is not accessible
   * @exception IllegalArgumentException  if argument problem
   * @exception InvocationTargetException if constructor excepted
   * @exception IOException               if I/O error in beans.instantiate
   */
  public static TypedObject createBean(ClassLoader cld, String className, TypedObject[] args) 
     throws ClassNotFoundException, NoSuchMethodException, InstantiationException, IllegalAccessException, 
            IllegalArgumentException, InvocationTargetException, IOException {
    if (args != null && args.length > 0) {
      // find the right constructor and use that to create bean
      Class cl = (cld != null) ? cld.loadClass (className): Class.forName (className);
      Class[] argTypes = new Class[args.length];
      Object[] argObjs = new Object[args.length];
      for (int i = 0; i < args.length; i++) {
        if (args[i] == null) continue;
        argTypes[i] = args[i].type;
        argObjs[i] = args[i].value;
        }
      
      // This needs to potentially check for convertable constructor
      // arguments as a second resort
      Constructor c = MathematicaMethodUtils.getConstructor(cl, argTypes);
      
      // We need to potentially convert argObjs before calling the constructor
      Object[] useArgs = new Object[args.length];
    
      Class[] cParams = c.getParameterTypes();
      int cParamSize = cParams.length;
      for (int i = 0; i < cParamSize; ++i) {
        try{
          useArgs[i] = MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(
            argTypes[i], argObjs[i], cParams[i]);
          }
        catch (MathematicaBSFException me) {
          useArgs[i] = argObjs[i];
          }
        }
      
      return TypedObjectFactory.create(cl, c.newInstance(useArgs));
    } else {
      // create the bean with no args constructor
      Object obj = Beans.instantiate(cld, className);
      return TypedObjectFactory.create(obj);
    }
  }

}
