
--[[

 This script should be copied into a file name "job_submit.lua"
 in the same directory as the SLURM configuration file, slurm.conf.

job_desc data structure defined in slurm/slurm.h, struct job_descriptor
part_list data structure src/slurmctld/slurmctld.h, struct part_record
job_rec data structure src/slurmctld/slurmctld.h, struct job_record

job_modify: see "man scontrol", job update for information about who
can change the various fields. root/admin can increate job time limit.
user can only decrease time limit. user can change very few fields.
I do not believe user can change their job partition.

--]]

function slurm_job_submit(job_desc, part_list, submit_uid)
    if job_desc.account == nil then
        slurm.log_user("--account option required")
        return slurm.ESLURM_INVALID_ACCOUNT
    end 

    if job_desc.time_limit == slurm.NO_VAL then
        slurm.log_user("--time limit option required")
        return slurm.ESLURM_INVALID_TIME_LIMIT
    end 

    if job_desc.qos == "expired" then
        slurm.log_user("Your allocation has expired")
        return slurm.ESLURM_INVALID_QOS
    end 

    if job_desc.partition == "buyin" then
        job_desc.partition = job_desc.account
    end 

    return slurm.SUCCESS
end

function slurm_job_modify(job_desc, job_rec, part_list, modify_uid)
    return slurm.SUCCESS
end

return slurm.SUCCESS
