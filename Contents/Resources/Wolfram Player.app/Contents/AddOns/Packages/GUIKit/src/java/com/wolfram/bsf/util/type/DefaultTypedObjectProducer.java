/*
 * @(#)DefaultTypedObjectProducer.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.type;

/**
 * DefaultTypedObjectProducer
 */
public class DefaultTypedObjectProducer implements TypedObjectProducer {

   public TypedObject create(Class c, Object o) {
     return new TypedObject(c, o);
     }
     
   public TypedObject[] createTypedArray(Object obj[]) {
    if (obj == null) return null;
    TypedObject result[] = new TypedObject[obj.length];
    for (int i = 0; i < obj.length; ++i)
      result[i] = (obj[i] != null ? new TypedObject(obj[i].getClass(), obj[i]) : null);
    return result;
    }
    
}

