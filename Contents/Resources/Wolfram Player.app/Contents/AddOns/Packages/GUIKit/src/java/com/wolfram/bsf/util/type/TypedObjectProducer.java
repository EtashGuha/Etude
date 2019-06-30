/*
 * @(#)TypedObjectProducer.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.type;

/**
 * TypedObjectProducer
 */
public interface TypedObjectProducer {

  public TypedObject create(Class c, Object o);
  
  public TypedObject[] createTypedArray(Object obj[]);
  
}

