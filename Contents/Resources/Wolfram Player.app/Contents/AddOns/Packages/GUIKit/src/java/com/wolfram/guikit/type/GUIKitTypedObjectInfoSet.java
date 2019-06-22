/*
 * @(#)GUIKitTypedObjectInfoSet.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.type;

import java.beans.BeanInfo;
import java.beans.PropertyDescriptor;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * GUIKitTypedObjectInfoSet
 */
public class GUIKitTypedObjectInfoSet  {

  private BeanInfo beanInfo = null;
  private Map propertyDescriptorCache = new WeakHashMap(3);
  
  public GUIKitTypedObjectInfoSet(BeanInfo info) {
    beanInfo = info;
    }

  public BeanInfo getBeanInfo() {return beanInfo;}
  
  public PropertyDescriptor getCachedPropertyDescriptor(String name) {
  	String lowerName = name.toLowerCase();
    Object propDes = propertyDescriptorCache.get(lowerName);
    if (propDes == null) {
      PropertyDescriptor descripts[] = beanInfo.getPropertyDescriptors();
      for(int i = 0; i < descripts.length; i++)
        if(lowerName.equalsIgnoreCase(descripts[i].getName())) {
          propDes = descripts[i];
          propertyDescriptorCache.put(lowerName, propDes);
          }
      }
    if (propDes != null) return (PropertyDescriptor)propDes;
    else return null;
    }
    
  public void clear() {
    propertyDescriptorCache.clear();
    propertyDescriptorCache = null;
    beanInfo = null;
    }
    
}

