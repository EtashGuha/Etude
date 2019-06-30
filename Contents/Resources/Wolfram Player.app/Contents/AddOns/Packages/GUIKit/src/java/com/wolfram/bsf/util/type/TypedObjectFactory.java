/*
 * @(#)TypedObjectFactory.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.type;

/**
 * TypedObjectFactory
 */
public class TypedObjectFactory {

  private static TypedObjectProducer producer = new DefaultTypedObjectProducer();
  
  public static TypedObject create(Class c, Object obj) {
    return producer.create(c, obj);
    }
    
  public static TypedObject create(Object obj) {
    return producer.create( obj != null ? obj.getClass() : Object.class, obj);
    }
    
  public static TypedObject[] createTypedArray(Object obj[]) {
    return producer.createTypedArray(obj);
    }
    
  public static TypedObjectProducer getProducer() {return producer;}
  public static void setProducer(TypedObjectProducer p) {
     producer = p;
     }
}

