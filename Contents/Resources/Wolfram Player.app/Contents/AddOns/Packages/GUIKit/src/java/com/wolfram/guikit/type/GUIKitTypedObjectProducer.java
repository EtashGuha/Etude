/*
 * @(#)GUIKitTypedObjectProducer.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.type;

import com.wolfram.bsf.util.type.TypedObject;
import com.wolfram.bsf.util.type.TypedObjectProducer;

/**
 * GUIKitTypedObjectProducer
 */
public class GUIKitTypedObjectProducer implements TypedObjectProducer {

   public TypedObject create(Class c, Object o) {
     return new GUIKitTypedObject(c, o);
     }
     
   public TypedObject[] createTypedArray(Object obj[]) {
    if (obj == null) return null;
    GUIKitTypedObject result[] = new GUIKitTypedObject[obj.length];
    for (int i = 0; i < obj.length; ++i)
      result[i] = (obj[i] != null ? new GUIKitTypedObject(obj[i].getClass(), obj[i]) : null);
    return result;
    }
    
}

