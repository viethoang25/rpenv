class AttackDeviceMsg:
    SUCCESSFUL = "100 - Successfully performed"
    SUCCESSFUL_COMPROMISED = "101 - Successfully performed and compromised the device"
    ATTACKED_HONEYPOT = "102 - Attacked a honeypot device"
    INBOUND_BLOCKED = "103 - Block by firewall inbound rules"
    SOURCE_NOT_FOUND = "104 - Cannot find any source devices to attack"
    UNAVAILABLE_TECHNIQUE = "105 - The used technique is not feasible"
    NOT_HAVE_PERMISSION = "106 - Do not have appropriate permissions"
    SUBNET_UNATTACKABLE = (
        "107 - Do not have information about the subnet of this device"
    )
    ALREADY_COMPROMISED = "108 - This device have already been compromised"
    TOO_DIFFICULT = "109 - Attacking failed because of the difficulity of the technique"


class AttackSubnetMsg:
    SUCCESSFUL = "201 - Successfully performed"
    UNEXPLOITABLE_SUBNET = "202 - Current subnet is not explorable"
    UNAVAILABLE_TECHNIQUE = "203 - The technique can be used for this subnet"
    ALREADY_USED_TECHNIQUE = "204 - The technique have been already used"
    TOO_DIFFICULT = "205 - Attacking failed because of the difficulity of the technique"


class Constants:
    ATTACK_DEVICE_MSG = AttackDeviceMsg()
    ATTACK_SUBNET_MSG = AttackSubnetMsg()
