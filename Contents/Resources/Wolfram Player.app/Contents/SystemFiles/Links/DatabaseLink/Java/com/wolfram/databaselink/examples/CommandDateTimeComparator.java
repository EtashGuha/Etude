package com.wolfram.databaselink.examples;

import java.util.Comparator;

public class CommandDateTimeComparator implements Comparator
{
  public int compare(Object a, Object b)
  {
    Command commandA = (Command)a;
    Command commandB = (Command)b;
    try
    {
      return new Long(commandA.getDateTime().getTime()).compareTo(new Long(commandB.getDateTime().getTime()));
    } catch (Exception e) {}
    return 0;
  }
}
